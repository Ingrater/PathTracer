module scene;

import modelloader;
import rendering;

import thBase.math3d.all;
import thBase.enumbitfield;
import thBase.format;
import thBase.allocator;
import thBase.math;
import thBase.logging;
import thBase.container.vector;
import thBase.allocator;
import thBase.io;
import thBase.algorithm;

import std.math;

class Scene
{
  alias void function(ref Material mat, const(char)[] materialName) MaterialFunc;

  static struct Node
  {
    Sphere sphere;
    union {
      Node*[2] childs;
      struct {
        void* dummy;
        Triangle* triangle;
      }
    }
  }

  static struct TriangleData
  {
    vec3 n0, n1, n2;
    Material* material;
    mat3 localToWorld;
  }

  Triangle[] m_triangles;
  TriangleData[] m_data;
  Node[] m_nodes;
  Material[] m_materials;
  Node* m_rootNode;

  this(const(char)[] path, MaterialFunc matFunc)
  {
    auto loader = New!ModelLoader();
    scope(exit) Delete(loader);

    loader.LoadFile(rcstring(path), Flags(ModelLoader.Load.Everything));

    size_t totalNumFaces = 0;
    foreach(ref mesh; loader.modelData.meshes)
    {
      totalNumFaces += mesh.faces.length;
    }

    m_materials = NewArray!Material(loader.modelData.materials.length);
    foreach(size_t i, ref material; m_materials)
    {
      matFunc(material, loader.modelData.materials[i].name);
    }

		m_triangles = NewArray!Triangle(totalNumFaces);
    m_data = NewArray!TriangleData(totalNumFaces);
    size_t currentFaceCount = 0;
    foreach(ref mesh; loader.modelData.meshes)
    {
      auto triangles = m_triangles[currentFaceCount..(currentFaceCount+mesh.faces.length)];
      auto data = m_data[currentFaceCount..(currentFaceCount+mesh.faces.length)];

      auto vertices = AllocatorNewArray!vec3(ThreadLocalStackAllocator.globalInstance, mesh.vertices.length);
      scope(exit) AllocatorDelete(ThreadLocalStackAllocator.globalInstance, vertices);

      auto normals = AllocatorNewArray!vec3(ThreadLocalStackAllocator.globalInstance, mesh.normals.length);
      scope(exit) AllocatorDelete(ThreadLocalStackAllocator.globalInstance, normals);

      const(ModelLoader.NodeDrawData*) findLeaf(const(ModelLoader.NodeDrawData*) node)
      {
        if(node.meshes.length > 0)
        {
          return node;
        }
        foreach(child; node.children)
        {
          auto result = findLeaf(child);
          if(result !is null)
          {
            return result;
          }
        }
        return null;
      }

      const(ModelLoader.NodeDrawData)* curNode = findLeaf(loader.modelData.rootNode);
      assert(curNode !is null, "no node with mesh found");
      mat4 transform = mat4.Identity().Right2Left();
      transform = loader.modelData.rootNode.transform * transform;
      while(curNode !is null && curNode != loader.modelData.rootNode)
      {
        transform = curNode.transform * transform;
        curNode = curNode.data.parent;
      }

      mat3 normalMatrix = transform.NormalMatrix();

      auto minBounds = vec3(float.max, float.max, float.max);
      auto maxBounds = vec3(-float.max, -float.max, -float.max);
      auto boundingRadius = 0.0f;

      foreach(size_t i, ref vertex; vertices)
      {
        vertex = transform * mesh.vertices[i];
        minBounds = minimum(minBounds, vertex);
        maxBounds = maximum(maxBounds, vertex);
        boundingRadius = max(boundingRadius, vertex.length);
      }
      logInfo("%s => minBounds %s, maxBounds %s", path, minBounds.f[], maxBounds.f[]);

      foreach(size_t i, ref normal; normals)
      {
        normal = normalMatrix * mesh.normals[i];
      }

		  foreach(size_t i,ref face; triangles)
      {			
			  face.v0 = vertices[mesh.faces[i].indices[0]];
        face.v1 = vertices[mesh.faces[i].indices[1]];
        face.v2 = vertices[mesh.faces[i].indices[2]];
        face.plane = Plane(face.v0, face.v1, face.v2);
		  }

      foreach(size_t i, ref d; data)
      {
        d.n0 = normals[mesh.faces[i].indices[0]];
        d.n1 = normals[mesh.faces[i].indices[1]];
        d.n2 = normals[mesh.faces[i].indices[2]];
        d.material = &m_materials[mesh.materialIndex];

        auto n = triangles[i].plane.normal;
        auto up = vec3(0,0,1);
        if(n.dot(up) > 0.9f)
        {
          up = vec3(0,1,0);
        }
        auto right = up.cross(n).normalize();
        up = n.cross(right).normalize();
        d.localToWorld = mat3(n, right, up);
      }
      currentFaceCount += mesh.faces.length;
    }

    /*uint nodesNeeded = 0;
    for(uint i=2; nodesNeeded < m_triangles.length; i*=2)
    {
      nodesNeeded += i;
    }*/
    m_nodes = NewArray!Node(m_triangles.length*2);

    uint nextNode = m_triangles.length;
    auto remainingNodes = ThreadLocalStackAllocator.globalInstance.AllocatorNew!(Vector!(Node*))();
    scope(exit) ThreadLocalStackAllocator.globalInstance.AllocatorDelete(remainingNodes);

    //fill the inital nodes
    remainingNodes.resize(m_triangles.length);
    foreach(size_t i, ref triangle; m_triangles)
    {
      Node *node = &m_nodes[i];
      remainingNodes[i] = node;
      auto centerPoint = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;      
      node.sphere.pos = centerPoint;
      node.sphere.radiusSquared = max(
                                      max((triangle.v0 - centerPoint).squaredLength, 
                                          (triangle.v1 - centerPoint).squaredLength),
                                      (triangle.v2 - centerPoint).squaredLength);
      assert(node.sphere.radiusSquared > 0.0f);
      node.dummy = null; //this means it is a leaf node
      node.triangle = &triangle;
    }

    //merge the nodes until there is only 1 node left
    size_t nodeToMerge = 0;
    while(remainingNodes.length > 1)
    {
      Node* nodeA = remainingNodes[nodeToMerge];
      auto centerPoint = nodeA.sphere.pos;

      //size_t smallestIndex = nodeToMerge + 1;
      size_t smallestIndex = (nodeToMerge) == 0 ? 1 : 0;
      float currentMinDistance = (remainingNodes[smallestIndex].sphere.pos - centerPoint).squaredLength;

      //foreach(size_t i, nodeB; remainingNodes[nodeToMerge+2..remainingNodes.length])
      foreach(size_t i, nodeB; remainingNodes.toArray())
      {
        if(i == nodeToMerge)
          continue;
        float dist = (nodeB.sphere.pos - centerPoint).squaredLength;
        if(dist < currentMinDistance)
        {
          currentMinDistance = dist;
          //smallestIndex = i + 2 + nodeToMerge;
          smallestIndex = i;
        }
      }

      Node* nodeB = remainingNodes[smallestIndex];
      assert(nodeA !is nodeB);

      Node* newNode = &m_nodes[nextNode++];
      float radiusA = nodeA.sphere.radius;
      float radiusB = nodeB.sphere.radius;
      vec3 rayThroughSpheres = (nodeB.sphere.pos - nodeA.sphere.pos).normalize();
      newNode.sphere.pos = ((nodeA.sphere.pos - (rayThroughSpheres * radiusA)) + (nodeB.sphere.pos + (rayThroughSpheres * radiusB))) * 0.5f;
      float newRadius = (radiusA + sqrt(currentMinDistance) + radiusB) * 0.5f;
      newNode.sphere.radiusSquared = newRadius * newRadius;
      newNode.childs[0] = nodeA;
      newNode.childs[1] = nodeB;

      assert(nodeToMerge != smallestIndex);
      remainingNodes[nodeToMerge] = newNode;
      remainingNodes.removeAtIndexUnordered(smallestIndex);

      nodeToMerge++;
      //if(nodeToMerge >= remainingNodes.length - 1)
      if(nodeToMerge >= remainingNodes.length)
        nodeToMerge = 0;
    }
    assert(remainingNodes.length == 1);
    m_rootNode = remainingNodes[0];

    static void countDepth(Node* node, size_t depth, ref size_t minDepth, ref size_t maxDepth, ref float sumRadius, ref float numRadii)
    {
      numRadii += 1.0f;
      sumRadius += node.sphere.radius;
      if(node.dummy is null)
      {
        minDepth = min(depth, minDepth);
        maxDepth = max(depth, maxDepth);
      }
      else
      {
        countDepth(node.childs[0], depth+1, minDepth, maxDepth, sumRadius, numRadii);
        countDepth(node.childs[1], depth+1, minDepth, maxDepth, sumRadius, numRadii);
      }
    }

    size_t minDepth = size_t.max;
    size_t maxDepth = 0;
    float numRadii = 0.0f;
    float sumRadius = 0.0f;
    countDepth(m_rootNode, 0, minDepth, maxDepth, sumRadius, numRadii);
    writefln("min-depth: %d, max-depth: %d, average-radius: %f", minDepth, maxDepth, sumRadius / numRadii);
  }

  ~this()
  {
    Delete(m_data);
    Delete(m_triangles);
    Delete(m_nodes);
    Delete(m_materials);
  }

  private bool traceHelper(const(Node*) node, ref const(Ray) ray, ref float rayPos, ref vec3 normal, ref const(TriangleData)* data) const
  {
    if(node.sphere.intersects(ray))
    {
      /*if(depth == 8)
      {
        float t = 0.0f;
        if(node.sphere.computeNearestIntersection(ray, t))
        {
          if(t > 0.0f && t < rayPos)
          {
            vec3 point = ray.get(t);
            rayPos = t;
            normal = (point - node.sphere.pos).normalize();
          }
          return true;
        }
        return false;
      }*/
      if(node.dummy is null)
      {
        //leaf node
        float pos = -1.0f;
        float u = 0.0f, v = 0.0f;
        if( node.triangle.intersects(ray, pos, u, v) && pos < rayPos && pos >= 0.0f )
        {
          auto n = node.triangle.plane.normal;
          if(n.dot(ray.dir) < 0)
          {
					  rayPos = pos;
            size_t index = cast(size_t)(node.triangle - m_triangles.ptr);
            const(TriangleData*) ldata = &m_data[index];

            float x = 1.0f / (u + v);
            float u1 = x * u;
            float v1 = x * v;
            const float sqrt2 = 1.414213562f;
            float d1 = sqrt((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
            float d2 = sqrt(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
            vec3 interpolated1 = ldata.n1 * d1 + ldata.n2 * d2;

            float len = sqrt(u1*u1 + v1*v1);
            float i1 = sqrt(u*u+v*v) / len;
            float i2 = 1.0f - i1;

            normal = ldata.n0 * i2 + interpolated1 * i1;
            data = ldata;
            return true;
          }
          return false;
        }
      }
      else
      {
        //non leaf node
        bool res1 = traceHelper(node.childs[0], ray, rayPos, normal, data);
        bool res2 = traceHelper(node.childs[1], ray, rayPos, normal, data);
        return res1 || res2;
      }
    }
    return false;
  }

	/**
  * Tests for a intersection with a already correctly transformed ray and this collision hull
  * Params:
  *  ray = the ray to test with
  *  rayPos = the position on the ray where it did intersect (out = result)
  *  normal = the normal at the intersection
  */
	bool trace(Ray ray, ref float rayPos, ref vec3 normal, ref const(TriangleData)* data) const {
		rayPos = float.max;
    debug {
      FloatingPointControl fpctrl;
      fpctrl.enableExceptions(FloatingPointControl.severeExceptions);
    }
		return traceHelper(m_rootNode, ray, rayPos, normal, data);
	}
}
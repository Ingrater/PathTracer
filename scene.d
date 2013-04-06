module scene;

import modelloader;

import thBase.math3d.all;
import thBase.enumbitfield;
import thBase.format;
import thBase.allocator;
import thBase.math;
import thBase.logging;
import thBase.container.vector;
import thBase.allocator;
import thBase.io;

import std.math;

class Scene
{
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

  Triangle[] m_triangles;
  Node[] m_nodes;
  Node* m_rootNode;

  this(const(char)[] path)
  {
    auto loader = New!ModelLoader();
    scope(exit) Delete(loader);

    loader.LoadFile(rcstring(path), Flags(ModelLoader.Load.Everything));

		if(loader.modelData.meshes.length > 1){
			throw New!RCException(format("The collision mesh '%s' does contain more that 1 mesh", path));
		}

		auto mesh = loader.modelData.meshes[0];
		m_triangles = NewArray!Triangle(mesh.faces.length);
    auto vertices = AllocatorNewArray!vec3(ThreadLocalStackAllocator.globalInstance, mesh.vertices.length);
    scope(exit) AllocatorDelete(ThreadLocalStackAllocator.globalInstance, vertices);

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
    //swap y and z axis so that z is up
    /*transform.f[5] = 0.0f;
    transform.f[6] = -1.0f;
    transform.f[10] = 0.0f;
    transform.f[9] = 1.0f;*/
    transform = loader.modelData.rootNode.transform * transform;
    while(curNode !is null && curNode != loader.modelData.rootNode)
    {
      transform = curNode.transform * transform;
      curNode = curNode.data.parent;
    }

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

		foreach(size_t i,ref face;m_triangles)
    {			
			face.v0 = vertices[mesh.faces[i].indices[0]];
      face.v1 = vertices[mesh.faces[i].indices[1]];
      face.v2 = vertices[mesh.faces[i].indices[2]];
      face.plane = Plane(face.v0, face.v1, face.v2);
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
    //TODO fix, needs to search all permutations NxN
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
      node.dummy = null; //this means it is a leaf node
      node.triangle = &triangle;
    }

    //merge the nodes until there is only 1 node left
    size_t nodeToMerge = 0;
    while(remainingNodes.length > 1)
    {
      Node* nodeA = remainingNodes[nodeToMerge];
      auto centerPoint = nodeA.sphere.pos;

      size_t smallestIndex = (nodeToMerge == 0) ? 1 : 0;
      float currentMinDistance = (remainingNodes[smallestIndex].sphere.pos - centerPoint).squaredLength;

      foreach(size_t i, nodeB; remainingNodes.toArray())
      {
        if(i == nodeToMerge)
          continue;
        float dist = (nodeB.sphere.pos - centerPoint).squaredLength;
        if(dist < currentMinDistance)
        {
          currentMinDistance = dist;
          smallestIndex = i;
        }
      }

      Node* nodeB = remainingNodes[smallestIndex];
      assert(nodeA !is nodeB);

      Node* newNode = &m_nodes[nextNode++];
      newNode.sphere.pos = (nodeA.sphere.pos + nodeB.sphere.pos) * 0.5f;
      float newRadius = nodeA.sphere.radius + sqrt(currentMinDistance) + nodeB.sphere.radius;
      newNode.sphere.radiusSquared = newRadius * newRadius;
      newNode.childs[0] = nodeA;
      newNode.childs[1] = nodeB;

      assert(nodeToMerge != smallestIndex);
      remainingNodes[nodeToMerge] = newNode;
      remainingNodes.removeAtIndexUnordered(smallestIndex);

      nodeToMerge++;
      if(nodeToMerge >= remainingNodes.length)
        nodeToMerge = 0;
    }
    assert(remainingNodes.length == 1);
    m_rootNode = remainingNodes[0];

    static void countDepth(Node* node, size_t depth, ref size_t minDepth, ref size_t maxDepth)
    {
      if(node.dummy is null)
      {
        minDepth = min(depth, minDepth);
        maxDepth = max(depth, maxDepth);
      }
      else
      {
        countDepth(node.childs[0], depth+1, minDepth, maxDepth);
        countDepth(node.childs[1], depth+1, minDepth, maxDepth);
      }
    }

    size_t minDepth = size_t.max;
    size_t maxDepth = 0;
    countDepth(m_rootNode, 0, minDepth, maxDepth);
    writefln("min-depth: %d, max-depth: %d", minDepth, maxDepth);
  }

  ~this()
  {
    Delete(m_triangles);
    Delete(m_nodes);
  }

  private static bool traceHelper(const(Node*) node, ref const(Ray) ray, ref float rayPos, ref vec3 normal)
  {
    if(node.sphere.intersects(ray))
    {
      if(node.dummy is null)
      {
        //leaf node
        float pos = -1.0f;
        if( node.triangle.intersects(ray, pos) && pos < rayPos && pos >= 0.0f )
        {
          auto n = node.triangle.plane.normal;
          //if(n.dot(ray.dir) < 0)
          {
					  rayPos = pos;
					  normal = n;
            return true;
          }
          return false;
        }
      }
      else
      {
        //non leaf node
        bool res1 = traceHelper(node.childs[0], ray, rayPos, normal);
        bool res2 = traceHelper(node.childs[1], ray, rayPos, normal);
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
	bool trace(Ray ray, ref float rayPos, ref vec3 normal) const {
		rayPos = float.max;
		return traceHelper(m_rootNode, ray, rayPos, normal);
	}
}
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
import thBase.container.octree;
import thBase.policies.hashing;
import thBase.allocator;
import thBase.io;
import thBase.algorithm;
import thBase.timer;
import thBase.casts;
import thBase.chunkfile;
import thBase.scoped;
import thBase.string;

import std.math;
import core.stdc.math;

//version = UseOctree;
//version = UseTopDown;

//Thread local bufferes for the tracing algorithm
const(Scene.Node)*[] g_readFrom;
const(Scene.Node)*[] g_writeTo;

static this()
{
  allocThreadLocals();
}

void allocThreadLocals()
{
  if(g_scene !is null && g_readFrom is null && g_writeTo is null)
  {
    g_readFrom = NewArray!(const(Scene.Node)*)(g_scene.m_nodes.length / 2);
    g_writeTo = NewArray!(const(Scene.Node)*)(g_scene.m_nodes.length / 2);
  }
}

static ~this()
{
  Delete(g_readFrom);
  Delete(g_writeTo);
}

class Scene
{
  alias void function(ref Material mat, const(char)[] materialName) MaterialFunc;

  static struct Node
  {
    Sphere sphere;
    bool same;
    union {
      Node*[2] childs;
      struct {
        void* dummy;
        Triangle* triangle;
      }
    }
  }

  static struct NodeOctreePolicy
  {
    static vec3 getPosition(Node* obj)
    {
      return obj.sphere.pos;
    }

    static AlignedBoxLocal getBoundingBox(Node* obj)
    {
      return AlignedBoxLocal(obj.sphere);
    }

    static bool hasMoved(Node* obj)
    {
      return false;
    }
  }

  // Data stored for each triangle
  static struct TriangleData
  {
    vec3 n0, n1, n2; // the normals at the three vertices of the triangle
    Material* material; // the material of the triangle
    mat3 localToWorld; // triangle space to world space transformation matrix
  }

  Triangle[] m_triangles;
  TriangleData[] m_data;
  Node[] m_nodes;
  Material[] m_materials;
  Node* m_rootNode;
  rcstring[] m_materialNames;

  this(const(char)[] path, MaterialFunc matFunc)
  {
    if(path.endsWith("tree", CaseSensitive.no))
    {
      loadTree(path, matFunc);
    }
    else
    {
      auto loader = New!ModelLoader();
      scope(exit) Delete(loader);

      loader.LoadFile(rcstring(path), Flags(ModelLoader.Load.Everything));

      auto leafs = New!(Vector!(const(ModelLoader.NodeDrawData)*))();
      leafs.reserve(16);
      scope(exit) Delete(leafs);

      void findLeafs(const(ModelLoader.NodeDrawData*) node)
      {
        if(node.meshes.length > 0)
        {
          leafs ~= node;
        }
        foreach(child; node.children)
        {
          findLeafs(child);
        }
      }

      findLeafs(loader.modelData.rootNode);
      assert(leafs.length > 0, "no leaf nodes found");

      size_t totalNumFaces = 0;
      foreach(leaf; leafs)
      {
        foreach(meshIndex; leaf.meshes)
        {
          auto mesh = &loader.modelData.meshes[meshIndex];
          totalNumFaces += mesh.faces.length;
        }
      }
      writefln("Scene has %d triangles", totalNumFaces);

      m_materials = NewArray!Material(loader.modelData.materials.length);
      m_materialNames = NewArray!rcstring(loader.modelData.materials.length);
      foreach(size_t i, ref material; m_materials)
      {
        m_materialNames[i] = loader.modelData.materials[i].name;
        matFunc(material, loader.modelData.materials[i].name);
      }

		  m_triangles = NewArray!Triangle(totalNumFaces);
      m_data = NewArray!TriangleData(totalNumFaces);
      auto minBounds = vec3(float.max, float.max, float.max);
      auto maxBounds = vec3(-float.max, -float.max, -float.max);
      auto boundingRadius = 0.0f;
      size_t currentFaceCount = 0;
      foreach(leaf; leafs)
      {
        auto curNode = leaf;
        mat4 transform = mat4.Identity().Right2Left();
        transform = loader.modelData.rootNode.transform * transform;
        while(curNode !is null && curNode != loader.modelData.rootNode)
        {
          transform = curNode.transform * transform;
          curNode = curNode.data.parent;
        }

        mat3 normalMatrix = transform.NormalMatrix();

        foreach(meshIndex; leaf.meshes)
        {
          auto mesh = &loader.modelData.meshes[meshIndex];
          auto triangles = m_triangles[currentFaceCount..(currentFaceCount+mesh.faces.length)];
          auto data = m_data[currentFaceCount..(currentFaceCount+mesh.faces.length)];

          auto vertices = AllocatorNewArray!vec3(ThreadLocalStackAllocator.globalInstance, mesh.vertices.length);
          scope(exit) AllocatorDelete(ThreadLocalStackAllocator.globalInstance, vertices);

          auto normals = AllocatorNewArray!vec3(ThreadLocalStackAllocator.globalInstance, mesh.normals.length);
          scope(exit) AllocatorDelete(ThreadLocalStackAllocator.globalInstance, normals);

          foreach(size_t i, ref vertex; vertices)
          {
            vertex = transform * mesh.vertices[i];
            minBounds = minimum(minBounds, vertex);
            maxBounds = maximum(maxBounds, vertex);
            boundingRadius = max(boundingRadius, vertex.length);
          }

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

            auto up = triangles[i].plane.normal;
            auto dir = vec3(1,0,0);
            if(abs(dir.dot(up)) > 0.9f)
            {
              dir = vec3(0,1,0);
            }
            auto right = up.cross(dir).normalize();
            dir = up.cross(right).normalize();
            d.localToWorld = mat3(dir, right, up);
          }
          currentFaceCount += mesh.faces.length;
        }
      }
      writefln("minBounds %s, maxBounds %s, boudingRadius %f", minBounds.f[], maxBounds.f[], boundingRadius);

      /*uint nodesNeeded = 0;
      for(uint i=2; nodesNeeded < m_triangles.length; i*=2)
      {
        nodesNeeded += i;
      }*/
      m_nodes = NewArray!Node(m_triangles.length*2);

      uint nextNode = m_triangles.length;
      auto remainingNodes = ThreadLocalStackAllocator.globalInstance.AllocatorNew!(Vector!(Node*))();
      scope(exit) ThreadLocalStackAllocator.globalInstance.AllocatorDelete(remainingNodes);

      auto timer = cast(shared(Timer))New!Timer();
      scope(exit) Delete(timer);

      auto startTime = Zeitpunkt(timer);

      Node* nodeFromTriangle(ref Triangle triangle)
      {
        Node *node = &m_nodes[nextNode++];
        auto centerPoint = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;      
        node.sphere.pos = centerPoint;
        node.same = false;
        node.sphere.radiusSquared = max(
                                        max((triangle.v0 - centerPoint).squaredLength, 
                                            (triangle.v1 - centerPoint).squaredLength),
                                        (triangle.v2 - centerPoint).squaredLength) + 0.01f;
        assert(node.sphere.radiusSquared > 0.0f);
        node.dummy = null; //this means it is a leaf node
        node.triangle = &triangle;
        return node;
      }

      Node* mergeNode(Node* nodeA, Node* nodeB)
      {
        if(nodeA is null && nodeB is null)
          return null;
        if(nodeA is null)
          return nodeB;
        if(nodeB is null)
          return nodeA;

        Node* newNode = null;

        Node* mergeHelper(Node* nodeA, Node* nodeB)
        {
          if(newNode is null)
          {
            assert(nodeA.sphere !in nodeB.sphere, "nodeA is inside nodeB");
            assert(nodeB.sphere !in nodeA.sphere, "nodeB is inside nodeA");
            newNode = &m_nodes[nextNode++];
            float radiusA = nodeA.sphere.radius;
            float radiusB = nodeB.sphere.radius;
            vec3 rayThroughSpheres = nodeB.sphere.pos - nodeA.sphere.pos;
            float dist = rayThroughSpheres.length;
            rayThroughSpheres = rayThroughSpheres.normalize();
            newNode.sphere.pos = ((nodeA.sphere.pos - (rayThroughSpheres * radiusA)) + (nodeB.sphere.pos + (rayThroughSpheres * radiusB))) * 0.5f;
            float newRadius = (radiusA + dist + radiusB) * 0.5f + 0.01f;
            newNode.sphere.radiusSquared = newRadius * newRadius;
          }
          newNode.childs[0] = nodeA;
          newNode.childs[1] = nodeB;
          newNode.same = false;
          assert(nodeA.sphere in newNode.sphere);
          assert(nodeB.sphere in newNode.sphere);
          return newNode;
        }

        static uint nodeDepth(Node* node, uint depth)
        {
          if(node.dummy is null)
            return depth;
          return max(nodeDepth(node.childs[0], depth+1),
                     nodeDepth(node.childs[1], depth+1));
        }

        Node* insertInto(Node* nodeToInsert, Node* boundingNode)
        {
          Node* currentNode = boundingNode;
          while(true)
          {
            if(currentNode.childs[0].dummy !is null && nodeToInsert.sphere in currentNode.childs[0].sphere)
            {
              currentNode = currentNode.childs[0];
              continue;
            }
            if(currentNode.childs[1].dummy !is null && nodeToInsert.sphere in currentNode.childs[1].sphere)
            {
              currentNode = currentNode.childs[1];
              continue;
            }
            break;
          }

          if(nodeDepth(currentNode.childs[0], 0) < nodeDepth(currentNode.childs[1], 0))
          {
            currentNode.childs[0] = mergeNode(currentNode.childs[0], nodeToInsert);
          }
          else
          {
            currentNode.childs[1] = mergeNode(currentNode.childs[1], nodeToInsert);
          }
          return boundingNode;
        }

        if(nodeA.sphere in nodeB.sphere)
        {
          if(nodeB.dummy is null)
          {
            newNode = &m_nodes[nextNode++];
            newNode.sphere = nodeB.sphere;
            nodeB.same = true;
          }
          else
          {
            //return nodeA.insertInto(nodeB);
            return insertInto(nodeA, nodeB);
          }
        }
        else if(nodeB.sphere in nodeA.sphere)
        {
          if(nodeA.dummy is null)
          {
            newNode = &m_nodes[nextNode++];
            newNode.sphere = nodeA.sphere;
            nodeA.same = true;
          }
          else
          {
            //return nodeB.insertInto(nodeA);
            return insertInto(nodeB, nodeA);
          }
        }
      
        return mergeHelper(nodeA, nodeB);
      }

      Node* bruteForceMerge(Vector!(Node*) nodesToMerge)
      {
        if(nodesToMerge.length == 0)
          return null;
        //merge the nodes until there is only 1 node left
        size_t nodeToMerge = 0;
        while(nodesToMerge.length > 1)
        {
          Node* nodeA = nodesToMerge[nodeToMerge];
          auto centerPoint = nodeA.sphere.pos;

          //size_t smallestIndex = nodeToMerge + 1;
          size_t smallestIndex = (nodeToMerge) == 0 ? 1 : 0;
          float currentMinDistance = (nodesToMerge[smallestIndex].sphere.pos - centerPoint).squaredLength;

          //foreach(size_t i, nodeB; nodesToMerge[nodeToMerge+2..nodesToMerge.length])
          foreach(size_t i, nodeB; nodesToMerge.toArray())
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

          Node* nodeB = nodesToMerge[smallestIndex];
          assert(nodeA !is nodeB);

          assert(nodeToMerge != smallestIndex);
          nodesToMerge[nodeToMerge] = mergeNode(nodeA, nodeB);
          nodesToMerge.removeAtIndexUnordered(smallestIndex);

          nodeToMerge++;
          //if(nodeToMerge >= nodesToMerge.length - 1)
          if(nodeToMerge >= nodesToMerge.length)
            nodeToMerge = 0;
        }
        assert(nodesToMerge.length == 1);
        return nodesToMerge[0];
      }


      //insert all inital nodes into the octree
      version(UseTopDown)
      {
        m_rootNode = mergeNode(nodeFromTriangle(m_triangles[0]), nodeFromTriangle(m_triangles[1]));
        foreach(ref triangle; m_triangles[2..$])
        {
          Node* newNode = nodeFromTriangle(triangle);
          if(!(newNode.sphere in m_rootNode.sphere))
          {
            m_rootNode = mergeNode(newNode, m_rootNode);
          }
          else
          {
            Node* insertInto = m_rootNode;
            while(true)
            {
              if(newNode.dummy is null)
                break;
              if(!(newNode.sphere in insertInto.childs[0]))
              {
              }
            }
          }
        }
      }
      else version(UseOctree)
      {
        alias LooseOctree!(Node*, NodeOctreePolicy, PointerHashPolicy, TakeOwnership.no) Octree;
        auto octree = New!Octree(boundingRadius * 2.0f, 0.1f);
        scope(exit) Delete(octree);
        foreach(size_t i, ref triangle; m_triangles)
        {
          Node *node = &m_nodes[i];
          auto centerPoint = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;      
          node.sphere.pos = centerPoint;
          node.sphere.radiusSquared = max(
                                          max((triangle.v0 - centerPoint).squaredLength, 
                                              (triangle.v1 - centerPoint).squaredLength),
                                          (triangle.v2 - centerPoint).squaredLength) + 0.01f;
          assert(node.sphere.radiusSquared > 0.0f);
          node.dummy = null; //this means it is a leaf node
          node.triangle = &triangle;
          octree.insert(node);
        }
        octree.optimize();

        auto stats = octree.ComputeStatistics();
        writefln("Octree minDepth %d, maxDepth %d, maxTriangles %d", stats.minDepth, stats.maxDepth, stats.maxNumObjects);

        Node* makeNode(Octree.Node octreeNode)
        {
          auto childs = octreeNode.childs;
          Node* result1 = null;
          if(childs.length > 0)
          {
            result1 = mergeNode(
              mergeNode(
                mergeNode(makeNode(childs[0]), makeNode(childs[1])),
                mergeNode(makeNode(childs[2]), makeNode(childs[3]))
              ),
              mergeNode(
                mergeNode(makeNode(childs[4]), makeNode(childs[5])),
                mergeNode(makeNode(childs[6]), makeNode(childs[7]))
              )
            );
          }

          Node* result2 = null;
          auto r = octreeNode.objects;
          if(!r.empty)
          {
            result2 = r.front;
            r.popFront();
            remainingNodes.resize(0);
            while(!r.empty)
            {
              remainingNodes ~= (r.front());
              r.popFront();
            }
            result2 = bruteForceMerge(remainingNodes);
          }

          return mergeNode(result1, result2);
        }
        m_rootNode = makeNode(octree.rootNode);
      }
      else
      {
        //fill the inital nodes
        remainingNodes.resize(m_triangles.length);
        foreach(size_t i, ref triangle; m_triangles)
        {
          Node *node = &m_nodes[i];
          remainingNodes[i] = node;
          auto centerPoint = (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;      
          node.sphere.pos = centerPoint;
          node.same = false;
          node.sphere.radiusSquared = max(
                                          max((triangle.v0 - centerPoint).squaredLength, 
                                              (triangle.v1 - centerPoint).squaredLength),
                                          (triangle.v2 - centerPoint).squaredLength) + 0.01f;
          assert(node.sphere.radiusSquared > 0.0f);
          node.dummy = null; //this means it is a leaf node
          node.triangle = &triangle;
        }
        m_rootNode = bruteForceMerge(remainingNodes);
      }

      auto endTime = Zeitpunkt(timer);

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
      writefln("Building tree took %f seconds", (endTime - startTime) / 1000.0f);
    }
    linearizeNodes();
  }

  ~this()
  {
    Delete(m_data);
    Delete(m_triangles);
    Delete(m_nodes);
    Delete(m_materials);
    Delete(m_materialNames);
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
    auto readFrom = g_readFrom;
    auto writeTo = g_writeTo;
    uint nextWrite = 0;
    uint numReads = 1;
    readFrom[0] = m_rootNode;

    bool result = false;

    while(numReads > 0)
    {
      foreach(node; readFrom[0..numReads])
      {
        if(node.same || node.sphere.intersects(ray))
        {
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
                float d1 = fastsqrt((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
                float d2 = fastsqrt(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
                vec3 interpolated1 = ldata.n1 * d1 + ldata.n2 * d2;

                float len = fastsqrt(u1*u1 + v1*v1);
                float i1 = fastsqrt(u*u+v*v) / len;
                float i2 = 1.0f - i1;

                normal = ldata.n0 * i2 + interpolated1 * i1;
                data = ldata;
                result = true;
              }
            }
          }
          else
          {
            //non leaf node
            writeTo[nextWrite++] = node.childs[0];
            writeTo[nextWrite++] = node.childs[1];
          }
        }
      }
      numReads = nextWrite;
      nextWrite = 0;
      swap(writeTo, readFrom);
    }

    return result;
  }

	/+bool trace(Ray ray, ref float rayPos, ref vec3 normal, ref const(TriangleData)* data) const {
		rayPos = float.max;
    debug {
      FloatingPointControl fpctrl;
      fpctrl.enableExceptions(FloatingPointControl.severeExceptions);
    }

    bool traceHelper(const(Node*) node)
    {
      if(node.same || node.sphere.intersects(ray))
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
              float d1 = sqrtf((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
              float d2 = sqrtf(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
              vec3 interpolated1 = ldata.n1 * d1 + ldata.n2 * d2;

              float len = sqrtf(u1*u1 + v1*v1);
              float i1 = sqrtf(u*u+v*v) / len;
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
          bool res1 = traceHelper(node.childs[0]);
          bool res2 = traceHelper(node.childs[1]);
          return res1 || res2;
        }
      }
      return false;
    }

		return traceHelper(m_rootNode);
	}+/

  /**
   * Computes the triangle index from a given triangle data
   */
  size_t getTriangleIndex(const(TriangleData*) data)
  {
    assert(data > m_data.ptr, "not a valid data pointer");
    return cast(size_t)(data - m_data.ptr);
  }

  /**
   * returns a array of all triangles
   */
  @property const(Triangle)[] triangles() const 
  {
    return m_triangles;
  }

  /**
   * returns a array of all triangle data
   */
  @property const(TriangleData)[] triangleData() const
  {
    return m_data;
  }

  /**
   * saves the internal tree structure to a file
   */
  void saveTree(const(char)[] filename)
  {
    auto outFile = scopedRef!Chunkfile(New!Chunkfile(rcstring(filename), Chunkfile.Operation.Write, Chunkfile.DebugMode.Off ));
    outFile.startWriting("tree", 1);
    scope(exit) outFile.endWriting();

    outFile.write(int_cast!uint(m_materials.length));
    foreach(ref name; m_materialNames)
    {
      outFile.write(int_cast!uint(name.length));
      outFile.write(name);
    }
    outFile.write(int_cast!uint(m_triangles.length));
    outFile.write(m_triangles);
    foreach(ref data; m_data)
    {
      outFile.write(data.n0);
      outFile.write(data.n1);
      outFile.write(data.n2);
      assert(data.material != null);
      uint materialIndex = int_cast!uint(data.material - m_materials.ptr);
      assert(materialIndex < m_materials.length);
      outFile.write(materialIndex);
      outFile.write(data.localToWorld);
    }

    outFile.write(int_cast!uint(m_nodes.length));
    foreach(ref node; m_nodes)
    {
      outFile.write(node.sphere);
      outFile.write(node.same);
      if(node.dummy is null)
      {
        outFile.write(cast(int)-1);
        if(node.triangle is null)
          outFile.write(cast(int)-1);
        else
          outFile.write(int_cast!int(node.triangle - m_triangles.ptr));
      }
      else
      {
        outFile.write(int_cast!int(node.childs[0] - m_nodes.ptr));
        outFile.write(int_cast!int(node.childs[1] - m_nodes.ptr));
      }
    }
    outFile.write(int_cast!int(m_rootNode - m_nodes.ptr));
  }

  private void loadTree(const(char)[] filename, MaterialFunc matFunc)
  {
    auto file = scopedRef!(Chunkfile)(New!Chunkfile(rcstring(filename), Chunkfile.Operation.Read));
    if(file.startReading("tree") != thResult.SUCCESS)
    {
      throw New!RCException(format("File '%s' is not a tree format", filename));
    }

    if(file.fileVersion != 1)
    {
      throw New!RCException(format("Tree '%s' does have old format, please reexport", filename));
    }

    // read materials
    uint numMaterials;
    file.read(numMaterials);
    m_materials = NewArray!Material(numMaterials);
    m_materialNames = NewArray!rcstring(numMaterials);
    foreach(size_t i, ref name; m_materialNames)
    {
      uint stringLength;
      file.read(stringLength);
      name = rcstring(stringLength);
      file.read(cast(char[])name[]);
      matFunc(m_materials[i], name[]);
    }

    // Read triangles
    uint numTriangles;
    file.read(numTriangles);
    m_triangles = NewArray!Triangle(numTriangles);
    file.read(m_triangles);

    m_data = NewArray!TriangleData(numTriangles);
    foreach(ref data; m_data)
    {
      file.read(data.n0);
      file.read(data.n1);
      file.read(data.n2);
      uint materialIndex;
      file.read(materialIndex);
      data.material = &m_materials[materialIndex];
      file.read(data.localToWorld);
    }

    // read nodes
    uint numNodes;
    file.read(numNodes);
    m_nodes = NewArray!Node(numNodes);
    foreach(ref node; m_nodes)
    {
      file.read(node.sphere);
      file.read(node.same);
      int firstIndex, secondIndex;
      file.read(firstIndex);
      file.read(secondIndex);
      if(firstIndex == -1)
      {
        //Leaf node
        node.dummy = null;
        if(secondIndex == -1)
          node.triangle = null;
        else
          node.triangle = &m_triangles[secondIndex];
      }
      else
      {
        node.childs[0] = &m_nodes[firstIndex];
        if(secondIndex == -1)
          node.childs[1] = null;
        else
          node.childs[1] = &m_nodes[secondIndex];
      }
    }
    int rootNodeIndex;
    file.read(rootNodeIndex);
    m_rootNode = &m_nodes[rootNodeIndex];

    file.endReading();
  }

  void linearizeNodes()
  {
    Node[] newNodes = NewArray!Node(m_nodes.length);
    size_t nextNode = 0;

    size_t[] nodeMapping = NewArray!size_t(m_nodes.length);

    uint findMaxDepth(Node* node, uint depth)
    {
      if(node.dummy is null)
        return depth;
      return max(findMaxDepth(node.childs[0], depth+1),
                 findMaxDepth(node.childs[1], depth+1));
    }

    uint maxDepth = findMaxDepth(m_rootNode, 0);

    void writeNodeAtDepth(Node* node, uint depth, uint currentDepth)
    {
      if(depth == currentDepth)
      {
        newNodes[nextNode] = *node;
        nodeMapping[node - m_nodes.ptr] = nextNode;
        nextNode++;
      }
      else if(node.dummy !is null)
      {
        writeNodeAtDepth(node.childs[0], depth, currentDepth + 1);
        writeNodeAtDepth(node.childs[1], depth, currentDepth + 1);
      }
    }

    for(uint i=0; i <= maxDepth; i++)
    {
      writeNodeAtDepth(m_rootNode, i, 0);
    }

    //patch nodes
    void patchNodePtr(ref Node* ptr)
    {
      if(ptr !is null)
      {
        auto oldIndex = ptr - m_nodes.ptr;
        auto newIndex = nodeMapping[oldIndex];
        ptr = &newNodes[newIndex];
      }
    }

    foreach(ref node; newNodes[0..nextNode])
    {
      if(node.dummy !is null)
      {
        patchNodePtr(node.childs[0]);
        patchNodePtr(node.childs[1]);
      }
    }

    Delete(m_nodes);
    Delete(nodeMapping);
    m_nodes = newNodes;
    m_rootNode = &m_nodes[0];
  }
}
module scene;

import modelloader;

import thBase.math3d.all;
import thBase.enumbitfield;
import thBase.format;
import thBase.allocator;
import thBase.math;
import thBase.logging;

class Scene
{
  Triangle[] m_triangles;

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
    mat4 transform = loader.modelData.rootNode.transform;
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
  }

	/**
  * Tests for a intersection with a already correctly transformed ray and this collision hull
  * Params:
  *  ray = the ray to test with
  *  rayPos = the position on the ray where it did intersect (out = result)
  *  normal = the normal at the intersection
  */
	bool intersects(Ray ray, ref float rayPos, ref vec3 normal) const {
		bool result = false;
		rayPos = float.max;
		foreach(ref triangle; m_triangles){
			float pos = -1.0f;
			if( triangle.intersects(ray, pos) ){
				if(pos < rayPos && pos >= 0.0f){
          auto n = triangle.plane.normal;
          if(n.dot(ray.dir) >= 0)
          {
            result = true;
					  rayPos = pos;
					  normal = n;
          }
				}
			}
		}
		return result;
	}
}
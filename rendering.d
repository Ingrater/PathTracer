module rendering;

import main;
import camera;
import scene;

import thBase.math3d.all;
import thBase.io;
import std.random;
import std.math;
import core.simd;
import core.stdc.math;

// global variables
__gshared Camera g_camera;
__gshared Scene g_scene;
__gshared uint g_width = 320;
__gshared uint g_height = 240;
__gshared uint g_numThreads = 8;

// Holds all information for a single pixel visible on the screen
struct Pixel
{
  vec3 color; //resulting pixel color. Each component has to be in the range [0,1]
  // Define additional per pixel values here
  float n = 0.0f;
  vec3 sum;
};

// Holds all information about a material
struct Material
{
  float emessive;
  vec3 color;
};

// Gets called on program shutdown
shared static ~this()
{
  Delete(g_camera);
  Delete(g_scene);
}


// Loads the scene and creates a camera
void loadScene()
{
  g_camera = New!Camera(30.0f, cast(float)g_height / cast(float)g_width);
  
  /*g_camera.setTransform(vec3(25, 10, 20), vec3(0, 0, 0), vec3(0, 0, 1));
  g_scene = New!Scene("teapot.thModel", &fillMaterial);*/

  g_camera.setTransform(vec3(-1, 26.5f, 10), vec3(0, 0, 9), vec3(0, 0, 1));
  g_scene = New!Scene("cornell-box.thModel", &fillMaterial);

  /*g_camera.setTransform(vec3(-1, 0, 7), vec3(0, 0, 7), vec3(0, 0, 1));
  g_scene = New!Scene("sponza2.thModel", &fillMaterial);*/

  /*g_camera.setTransform(vec3(3, 3, 3), vec3(0, 0, 0), vec3(0, 0, 1));
  g_scene = New!Scene("chest1.thModel";)*/
}

// Called for each material found in the model file
void fillMaterial(ref Material mat, const(char)[] materialName)
{
  mat.color.x = 1.0f;
  mat.color.y = 1.0f;
  mat.color.z = 1.0f;
  mat.emessive = 0.0f;
  if(materialName == "Light")
  {
    mat.emessive = 10.0f;
  }
  else if(materialName == "Red")
  {
    mat.color.x = 1.0f;
    mat.color.y = 0.0f;
    mat.color.z = 0.0f;
  }
  else if(materialName == "Green")
  {
    mat.color.x = 0.0f;
    mat.color.y = 1.0f;
    mat.color.z = 0.0f;
  }
}

// Computes a view ray for a given pixel index
Ray getViewRay(uint pixelIndex, ref Random gen)
{
  float pixelSizeX = 1.0f / cast(float)g_width;
  float pixelSizeY = 1.0f / cast(float)g_height;
  float x = cast(float)(pixelIndex % g_width) / cast(float)g_width * 2.0f - 1.0f + uniform(-pixelSizeX, pixelSizeX, gen);
  float y = cast(float)(pixelIndex / g_width) / cast(float)g_height * 2.0f - 1.0f + uniform(-pixelSizeY, pixelSizeY, gen);
  return g_camera.getScreenRay(x, y);
}

float haltonSequence(float index, float base)
{
  float result = 0;
  float f = 1.0f / base;
  while(index > 0.0f)
  {
    result = result + f * (core.stdc.math.fmod(index, base));
    index = floorf(index / base);
    f = f / base;
  }
  return result;
}

/*
psi = 0..360
phi = 0..90

(cos(psi) -sin(psi)   0)     (cos(phi))     (cos(psi) * cos(phi))
(sin(psi)  cos(psi)   0)  *  (0       )  =  (sin(psi) * cos(phi))
(       0         0   1)     (sin(phi))  =  (sin(phi)           )
*/

vec3 angleToLocalDirection(float phi, float psi)
{
  float cosPhi = cosf(phi);
  auto result = vec3(cosf(psi) * cosPhi, sinf(psi) * cosPhi, sinf(phi));
  return result;
}

vec3 angleToDirection(float phi, float psi, ref const(vec3) normal)
{
  float cosPhi = cosf(phi);
  auto localDir = vec3(cosf(psi) * cosPhi, sinf(psi) * cosPhi, sinf(phi));

  auto up = normal;
  auto dir = vec3(1,0,0);
  if(abs(dir.dot(up)) > 0.9f)
  {
    dir = vec3(0,1,0);
  }
  auto right = up.cross(dir).normalize();
  dir = up.cross(right).normalize();
  return mat3(dir, right, up) * localDir;
}

/+void computeL(uint pixelIndex, ref Pixel pixel, ref Random gen)
{
  Ray viewRay = getViewRay(pixelIndex);
  float rayPos = 0.0f;
  vec3 normal;
  const(Scene.TriangleData)* data;
  if( g_scene.trace(viewRay, rayPos, normal, data)){
    if(data.material.emessive > 0.0f)
    {
      pixel.color.x = pixel.color.y = pixel.color.z = 1.0f;
    }
    else
    {
      float NdotL = abs(normal.dot(-viewRay.dir));
      pixel.color = data.material.color * NdotL;
    }
    /*float NdotL = abs(normal.dot(-viewRay.dir));
    pixel.color.x = NdotL;
    pixel.color.y = NdotL;
    pixel.color.z = NdotL;*/
  }
  else
  {
    pixel.color = vec3(0,0,0);
  }
}+/

enum float BRDF = 1.0f / PI;

void computeL(uint pixelIndex, ref Pixel pixel, ref Random gen)
{
	enum uint N = 10;
	for(uint i=0; i<N; i++){
		Ray viewRay = getViewRay(pixelIndex, gen);
		float rayPos = 0.0f;
		vec3 normal;
		const(Scene.TriangleData)* data;
		if( g_scene.trace(viewRay, rayPos, normal, data)){
			vec3 hitPos = viewRay.get(rayPos);
			pixel.sum += computeL(hitPos, -viewRay.dir, normal, data, gen, 0);
		}
		else{
		  pixel.color.x = pixel.color.y = pixel.color.z = 0.0f;
		}
	}
	pixel.n += N;
	pixel.color = pixel.sum / pixel.n;
}

vec3 computeL(vec3 pos, vec3 theta, ref const(vec3) normal, const(Scene.TriangleData)* data, ref Random gen, uint depth)
{
	if(depth > 2) return data.material.emessive * data.material.color; 
	float psi = uniform(0, 2 * PI, gen);
	float phi = uniform(0, PI_2, gen);
	vec3 sampleDir = angleToDirection(phi, psi, normal);
	Ray sampleRay = Ray(pos, sampleDir);
	float hitDistance = 0.0f;
	vec3 hitNormal;
	const(Scene.TriangleData)* hitData;
	if( g_scene.trace(sampleRay, hitDistance, hitNormal, hitData)){
		vec3 hitPos = sampleRay.get(hitDistance);
		return (data.material.emessive * data.material.color) + 
      (BRDF * PI * data.material.color * computeL(hitPos, -sampleDir, hitNormal, hitData, gen, depth + 1));
	}
	return data.material.emessive * data.material.color; 
}

/**
* computes the output color of the generated image
* 
* Params: 
*  pixelOffset = the pixel offset (for multithreading)
*  pixels = the pixels that should be computed
*  gen = the random number generator
*/
void computeOutputColor(uint pixelOffset, Pixel[] pixels, ref Random gen)
{
  foreach(uint pixelIndex, ref pixel; pixels)
  {
    computeL(pixelOffset + pixelIndex, pixel, gen);
  }                          
}
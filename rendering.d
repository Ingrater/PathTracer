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
  /*debug {
    FloatingPointControl fpctrl;
    fpctrl.enableExceptions(FloatingPointControl.severeExceptions);
  }*/
  foreach(uint pixelIndex, ref pixel; pixels)
  {
    Ray viewRay = getViewRay(pixelOffset + pixelIndex, gen);
    float rayPos = 0.0f;
    vec3 normal;
    const(Scene.TriangleData)* data;
    if( g_scene.trace(viewRay, rayPos, normal, data))
    {
      vec3 hitPos = viewRay.get(rayPos);

      auto e = vec3(0.0f, 0.0f, 0.0f);
      enum uint N = 9;
      //naive sampling
      for(uint i=0; i<N; i++)
      {
        float psi = uniform(0.0f, PI * 2.0f, gen);
        float phi = uniform(0.0f, PI_2, gen);
        auto outDir = angleToLocalDirection(phi, psi);
        outDir = data.localToWorld * outDir;
        e += evalRenderingEquation(outDir, hitPos, normal, data, gen, 0);
      }

      // grid sampling
      /*
      immutable(float) step = 1.0f / 3.0f;
      for(uint x=0; x<3; x++)
      {
        for(uint y=0; y<3; y++)
        {
          float psi = uniform(x * step * PI * 2.0f, (x+1) * step * PI * 2.0f, gen);
          float phi = uniform(y * step * PI_2, (y+1) * step * PI_2, gen);
          auto outDir = angleToLocalDirection(phi, psi);
          outDir = data.localToWorld * outDir;
          e += evalRenderingEquation(outDir, hitPos, normal, data, gen, 0);
        }
      }*/

      //halton sampling
      /*for(uint i=0; i<N; i++)
      {
        float psi = haltonSequence(pixel.n + i, 2.0f) * PI * 2.0f;
        float phi = haltonSequence(pixel.n + i, 3.0f) * PI_2;
        auto outDir = angleToLocalDirection(phi, psi);
        outDir = data.localToWorld * outDir;
        e += evalRenderingEquation(outDir, hitPos, normal, data, gen, 0);
      }*/
      
      pixel.n += cast(float)N;
      pixel.sum += e;
      e = pixel.sum / pixel.n;

      pixel.color.x = e.x;
      pixel.color.y = e.y;
      pixel.color.z = e.z;
      /*if(data.material.emessive > 0.0f)
      {
        pixel.color.x = pixel.color.y = pixel.color.z = 1.0f;
      }
      else
      {
        float NdotL = abs(normal.dot(-viewRay.dir));
        pixel.color.x = data.material.color.x * NdotL;
        pixel.color.y = data.material.color.y * NdotL;
        pixel.color.z = data.material.color.z * NdotL;
      }*/
      /*float NdotL = abs(normal.dot(-viewRay.dir));
      pixel.color.x = NdotL;
      pixel.color.y = NdotL;
      pixel.color.z = NdotL;*/
    }
    else
    {
      pixel.color.x = pixel.color.z = 1.0f;
      pixel.color.y = 0.0f;
    }
  }                          
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

vec3 evalRenderingEquation(ref const(vec3) dir, ref const(vec3) pos, ref const(vec3) normal, const(Scene.TriangleData)* data, ref Random gen, uint depth)
{
  const(float) BRDF = 1.0f / (PI); //* 2.0f);
  if(depth > 3 || data.material.emessive > 0.0f)// || uniform(0.0f, 1.0f, gen) > BRDF * 2.0f)
  {
    //writefln("exit at depth %d", depth);
    return data.material.emessive * data.material.color; 
  }
  auto outRay = Ray(pos + normal * 0.001f, dir);

  //trace into the scene
  float rayPos = 0.0f;
  vec3 hitNormal;
  const(Scene.TriangleData)* hitData;
  if( g_scene.trace(outRay, rayPos, hitNormal, hitData))
  {
    auto hitPos = outRay.get(rayPos);

    /*float index = cast(float)uniform(0, 900, gen);
    float psi = haltonSequence(index, 2.0f) * PI * 2.0f;
    float phi = haltonSequence(index, 3.0f) * PI_2;*/

    float psi = uniform(0.0f, PI * 2.0f, gen);
    float phi = uniform(0.0f, PI_2, gen);
    auto outDir = angleToLocalDirection(phi, psi);
    outDir = hitData.localToWorld * outDir;

    
    auto result = (evalRenderingEquation(outDir, hitPos, hitNormal, hitData, gen, depth + 1) * BRDF * data.material.color /* normal.dot(dir)*/) * PI + data.material.emessive * data.material.color;
    //assert(result >= 0.0f);
    return result;
  }
  //writefln("exit at depth %d", depth);
  return data.material.emessive * data.material.color; 
}
module rendering;

import main;
import camera;
import scene;

import thBase.math3d.all;
import std.random;
import std.math;
import core.simd;

// global variables
__gshared Camera g_camera;
__gshared Scene g_scene;
__gshared uint g_width = 320;
__gshared uint g_height = 240;
__gshared uint g_numThreads = 8;

struct Color
{
  float r = 0.0f, g = 0.0f, b = 0.0f;
}

struct Pixel
{
  Color color;
  //Additional per pixel variables follow here
  float n = 0.0f;
  float sum = 0.0f;
};

struct Material
{
  float emessive;
  Color color;
};

shared static ~this()
{
  Delete(g_camera);
  Delete(g_scene);
}



void loadScene()
{
  g_camera = New!Camera(30.0f, cast(float)g_height / cast(float)g_width);
  /*g_camera.setTransform(vec3(20, 5, 20), vec3(0, 0, 0), vec3(0, 0, 1));
  g_scene = New!Scene("teapot.thModel");*/
  g_camera.setTransform(vec3(-1, 26.5f, 10), vec3(0, 0, 9), vec3(0, 0, 1));
  g_scene = New!Scene("cornell-box.thModel", &fillMaterial);
  /*g_camera.setTransform(vec3(3, 3, 3), vec3(0, 0, 0), vec3(0, 0, 1));
  g_scene = New!Scene("chest1.thModel";)*/
}

void fillMaterial(ref Material mat, const(char)[] materialName)
{
  mat.color.r = 1.0f;
  mat.color.g = 1.0f;
  mat.color.b = 1.0f;
  mat.emessive = 0.0f;
  if(materialName == "Light")
  {
    mat.emessive = 100.0f;
  }
  else if(materialName == "Red")
  {
    mat.color.r = 1.0f;
    mat.color.g = 0.0f;
    mat.color.b = 0.0f;
  }
  else if(materialName == "Green")
  {
    mat.color.r = 0.0f;
    mat.color.g = 1.0f;
    mat.color.b = 0.0f;
  }
}

Ray getViewRay(uint pixelIndex)
{
  float x = cast(float)(pixelIndex % g_width) / cast(float)g_width * 2.0f - 1.0f;
  float y = cast(float)(pixelIndex / g_width) / cast(float)g_height * 2.0f - 1.0f;
  return g_camera.getScreenRay(x, y);
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
    Ray viewRay = getViewRay(pixelOffset + pixelIndex);
    float rayPos = 0.0f;
    vec3 normal;
    const(Scene.TriangleData)* data;
    if( g_scene.trace(viewRay, rayPos, normal, data))
    {
      vec3 hitPos = viewRay.get(rayPos);

      float e = 0.0f;
      for(uint i=0; i<5; i++)
      {
        e += evalRenderingEquation(hitPos, normal, data, gen, 0);
      }
      pixel.n += 5.0f;
      pixel.sum += e;
      e = pixel.sum * PI / pixel.n;

      pixel.color.r = pixel.color.g = pixel.color.b = e;
      /*if(data.material.emessive > 0.0f)
      {
        pixel.color.r = pixel.color.g = pixel.color.b = 1.0f;
      }
      else
      {
        float NdotL = abs(normal.dot(-viewRay.dir));
        pixel.color.r = data.material.color.r * NdotL;
        pixel.color.g = data.material.color.g * NdotL;
        pixel.color.b = data.material.color.b * NdotL;
      }*/
    }
    else
    {
      pixel.color.g = pixel.color.b = 0.0f;
      pixel.color.r = 1.0f;
    }
  }                          
}

vec3 angleToLocalDirection(float phi, float psi)
{
  float cosPsi = cos(psi);
  auto result = vec3(cos(phi) * cosPsi, sin(phi) * cosPsi, sin(psi));
  return result;
}

float evalRenderingEquation(ref const(vec3) pos, ref const(vec3) normal, const(Scene.TriangleData)* data, ref Random gen, uint depth)
{
  const(float) BRDF = 1.0f / (PI * 2.0f);
  if(depth >= 1)
  {
    return data.material.emessive;
  }

  float phi = uniform(0.0f, PI * 2.0f, gen);
  float psi = uniform(0.0f, PI_2, gen);
  auto outDir = angleToLocalDirection(phi, psi);
  outDir = data.localToWorld * outDir;
  auto outRay = Ray(pos, outDir);

  //trace into the scene
  float rayPos = 0.0f;
  vec3 hitNormal;
  const(Scene.TriangleData)* hitData;
  if( g_scene.trace(outRay, rayPos, hitNormal, hitData))
  {
    auto hitPos = outRay.get(rayPos);
    auto result = evalRenderingEquation(hitPos, hitNormal, hitData, gen, depth + 1) * BRDF * normal.dot(outDir) + data.material.emessive;
    //assert(result >= 0.0f);
    return result;
  }
  return data.material.emessive;
}
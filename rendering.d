module rendering;

import main;
import camera;
import scene;

import thBase.math3d.all;
import std.random;
import std.math;
import core.simd;

struct Color
{
  float r,g,b;
}

struct Pixel
{
  Color color;
  //Additional per pixel variables follow here
};

__gshared Camera g_camera;
__gshared Scene g_scene;

shared static ~this()
{
  Delete(g_camera);
  Delete(g_scene);
}



void loadScene()
{
  g_camera = New!Camera(45.0f, cast(float)g_height / cast(float)g_width);
  g_camera.setTransform(vec3(20, 0, 20), vec3(0, 0, 0), vec3(0, 0, 1));
  g_scene = New!Scene("teapot.thModel");
  /*g_camera.setTransform(vec3(3, 3, 3), vec3(0, 0, 0), vec3(0, 0, 1));
  g_scene = New!Scene("chest1.thModel";)*/
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
  {
    Ray testRay = g_camera.getScreenRay(0.33125f, 0.1833333f);
    float rayPos = 0.0f;
    vec3 faceNormal;
    bool hits = g_scene.trace(testRay, rayPos, faceNormal);
  }


  foreach(uint pixelIndex, ref pixel; pixels)
  {
    Ray viewRay = getViewRay(pixelOffset + pixelIndex);
    float rayPos = 0.0f;
    vec3 faceNormal;
    if( g_scene.trace(viewRay, rayPos, faceNormal))
    {
      pixel.color.r = pixel.color.g = pixel.color.b = abs(faceNormal.dot(-viewRay.dir));
    }
    else
    {
      pixel.color.g = pixel.color.b = 0.0f;
      pixel.color.r = 1.0f;
    }
  }                          
}

module rendering;

import main;
import thBase.math3d.all;

struct Pixel
{
  float r,g,b;
  float n; //how many rays have been summed
};

shared static this()
{
}

shared static ~this()
{
}

Ray getViewRay(uint pixelIndex)
{
  float x = cast(float)(pixelIndex % g_width) / cast(float)g_width * 2.0f;
  float y = cast(float)(pixelIndex / g_width) / cast(float)g_height * 2.0f;
}

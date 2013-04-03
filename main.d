module main;

import thBase.io;
import sdl;

void setPixel(SDL.Surface *screen, int x, int y, ubyte r, ubyte g, ubyte b)
{
  uint *pixmem32;
  uint colour;  

  colour = SDL.MapRGB( screen.format, r, g, b );

  pixmem32 = cast(uint*)(screen.pixels + (y * screen.pitch + x * 4));
  *pixmem32 = colour;
}

void drawScreen(SDL.Surface* screen, int h)
{
  for(int y = 0; y < screen.height; y++ ) 
  {
    for(int x = 0; x < screen.width; x++ ) 
    {
      setPixel(screen, x, y, cast(ubyte)((x*x)/256+3*y+h), cast(ubyte)((y*y)/256+x+h), cast(ubyte)h);
    }
  }

  SDL.Flip(screen); 
}

__gshared uint g_width = 640;
__gshared uint g_height = 480;

int main(string[] argv)
{
  SDL.LoadDll("SDL.dll","libSDL-1.2.so.0");

  if(SDL.Init(SDL.INIT_VIDEO) < 0)
  {
    writefln("Initializing SDL failed");
    return 1;
  }
  scope(exit) SDL.Quit();

  SDL.Surface* screen = SDL.SetVideoMode(g_width, g_height, 32, SDL.HWSURFACE);

  if(screen is null)
  {
    writefln("creating screen surface failed");
    return 1;
  }

  int h = 0;
  bool run = true;
  SDL.Event event;

  while(run)
  {
    drawScreen(screen,h++);
    while(SDL.PollEvent(&event)) 
    {      
      switch (event.type) 
      {
        case SDL.QUIT:
          run = false;
          break;
        case SDL.KEYDOWN:
          run = false;
          break;
        default:
      }
    }
  }

  return 0;
}

module trianglehelpers;

import thBase.math3d.vecs;
import std.math;
import thBase.math;

auto interpolate(T)(float u, float v, T val0, T val1, T val2)
{
  if(u < 0.0f)
    u = 0.0f;
  if(v < 0.0f)
    v = 0.0f;
  float uv = (u + v);
  float u1, v1;
  if(uv != 0.0f)
  {
    float x = 1.0f / uv;
    u1 = x * u;
    v1 = x * v;
  }
  else
  {
    u1 = u;
    v1 = v;
  }
  immutable float sqrt2 = 1.414213562f;
  float d1 = sqrt((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
  float d2 = sqrt(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
  auto interpolated1 = val1 * d1 + val2 * d2;

  float i1;
  if(uv != 0.0f)
  {
    float len = sqrt(u1*u1 + v1*v1);
    i1 = sqrt(u*u+v*v) / len; 
  }
  else
  {
    i1 = 0.0f;
  }
  float i2 = 1.0f - i1;

  return val0 * i2 + interpolated1 * i1;
}

auto extrapolate(T)(float u, float v, T val0, T val1, T val2)
{
  float uv = (u + v);
  float u1, v1;
  if(uv != 0.0f)
  {
    float x = 1.0f / uv;
    u1 = x * u;
    v1 = x * v;
  }
  else
  {
    u1 = u;
    v1 = v;
  }
  immutable float sqrt2 = 1.414213562f;
  float d1 = sqrt((1.0f-u1)*(1.0f-u1) + v1*v1) / sqrt2;
  float d2 = sqrt(u1*u1 + (1.0f-v1)*(1.0f-v1)) / sqrt2;
  auto interpolated1 = val1 * d1 + val2 * d2;

  float i1;
  if(uv != 0.0f)
  {
    float len = sqrt(u1*u1 + v1*v1);
    i1 = sqrt(u*u+v*v) / len; 
  }
  else
  {
    i1 = 0.0f;
  }
  float i2 = 1.0f - i1;

  return val0 * i2 + interpolated1 * i1;
}

vec2 computeUV(vec2 val, vec2 v0, vec2 v1, vec2 v2)
{
  auto p = v0;
  auto r1 = (v1 - v0);
  auto r2 = (v2 - v0);
  float d = (r1.x * r2.y - r1.y * r2.x);
  if(d.epsilonCompare(0.0f))
  {
    return vec2(float.max, float.max);
  }
  float u = (-p.x * r2.y + p.y * r2.x - r2.x * val.y + r2.y * val.x) / d;
  float v = ( p.x * r1.y - p.y * r1.x + r1.x * val.y - r1.y * val.x) / d;
  return vec2(u, v);
}
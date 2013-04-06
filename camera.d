module camera;

import std.math;
import thBase.math3d.vecs;
import thBase.math3d.mats;
import thBase.math3d.ray;

class Camera
{
  private:
    float m_tanFov, m_aspectRatio;
    vec3 m_dir, m_up, m_right, m_pos;

  public:
    // fov = field of view in degrees
    this(float fov, float aspectRatio)
    {
      m_tanFov = tan(fov / 180.0f * PI);
      m_aspectRatio = aspectRatio;
    }

    void setTransform(vec3 from, vec3 to, vec3 up)
    {
      m_pos = from;

      m_dir = (to - from).normalize();
      m_right = up.cross(m_dir).normalize();
      m_up = m_right.cross(m_dir).normalize();
    }

    /**
     * returns a ray through the screen at position x,y
     * 
     * Params:
     *  x = x in screen space [-1, 1]
     *  y = y in screen space [-1, 1]
     */
    Ray getScreenRay(float x, float y)
    {
      vec3 rayDir = (m_dir + m_right * x * m_tanFov + m_up * y * m_tanFov * m_aspectRatio).normalize();
      return Ray(m_pos, rayDir);
    }
}
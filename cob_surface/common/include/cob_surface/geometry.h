/*!
 *****************************************************************
 * \file
 *
 * \note
 *   Copyright (c) 2014 \n
 *   Fraunhofer Institute for Manufacturing Engineering
 *   and Automation (IPA) \n\n
 *
 *****************************************************************
 *
 * \note
 *  Project name: care-o-bot
 * \note
 *  ROS stack name: cob_environment_perception
 * \note
 *  ROS package name: cob_surface
 *
 * \author
 *  Author: Steffen Fuchs, email:richard.bormann@ipa.fhg.de
 * \author
 *  Supervised by: Richard Bormann, email:richard.bormann@ipa.fhg.de
 *
 * \date Date of creation: 12/2014
 *
 * \brief
 * Description:
 *
 * ToDo:
 *
 *
 *****************************************************************
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     - Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer. \n
 *     - Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution. \n
 *     - Neither the name of the Fraunhofer Institute for Manufacturing
 *       Engineering and Automation (IPA) nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission. \n
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License LGPL as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License LGPL for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License LGPL along with this program.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

#ifndef COB_SURFACE_GEOMETRY_H
#define COB_SURFACE_GEOMETRY_H

/* This header defines all policies necessary for using
 * the surface class with several algorithms
 */

#include "cob_surface/surface.h"

namespace cob_surface
{
  namespace Geometry
  {
    /** 
     * compute the barycentric weights of a point p on triangle a,b,c
     * 
     * @param p - query point
     * @param a - point of triangle
     * @param b - point of triangle
     * @param c - point of triangle
     * @param u - out: weight of a
     * @param v - out: weight of b
     * @param w - out: weight of c
     */
    template<typename ScalarT, typename PointT>
    void barycentric2d(const PointT& p, 
                       const PointT& a,
                       const PointT& b,
                       const PointT& c,
                       ScalarT &u,
                       ScalarT &v,
                       ScalarT &w)
    {
      PointT v0 = b - a;
      PointT v1 = c - a;
      PointT v2 = p - a;
      ScalarT d00 = v0.dot(v0);
      ScalarT d01 = v0.dot(v1);
      ScalarT d11 = v1.dot(v1);
      ScalarT d20 = v2.dot(v0);
      ScalarT d21 = v2.dot(v1);
      ScalarT denom_inv = 1./ (d00 * d11 - d01 * d01);
      v = (d11 * d20 - d01 * d21) * denom_inv;
      w = (d00 * d21 - d01 * d20) * denom_inv;
      u = 1. - v - w;
    }

    /** 
     * compute the intersection of two lines a1->a2 and b1->b2
     * 
     * @param a1 - tip of line a
     * @param a2 - foot of line a
     * @param b1 - tip of line b
     * @param b2 - foot of line b
     * @param x - out: intersection point if any
     * @param t - out: x = a2 + t * (a1 - a2)
     * @param s - out: x = b2 + s * (b1 - b2)
     * 
     * @return false if non intersecting
     */
    template<typename ScalarT, typename PointT>
    bool lineLineIntersection(const PointT& a1, const PointT& a2,
                              const PointT& b1, const PointT& b2,
                              PointT& x, ScalarT& t, ScalarT& s)
    {
      PointT sa = a1 - a2;
      PointT sb = b1 - b2;
      ScalarT denom = sa[0]*sb[1] - sb[0]*sa[1];
      if (denom == 0) return false;  // Collinear
      bool denomPositive = denom > 0;

      PointT sab = a2 - b2;
      ScalarT s_numer = sa[0]*sab[1] - sa[1]*sab[0];
      if ((s_numer < 0) == denomPositive) return false; // No collision

      ScalarT t_numer = sb[0]*sab[1] - sb[1]*sab[0];
      if ((t_numer < 0) == denomPositive) return false; // No collision

      if (((s_numer > denom) == denomPositive) || ((t_numer > denom) == denomPositive))
        return false; // No collision

      t = t_numer / denom;
      s = s_numer / denom;
      x = a2 + t * sa;
      return true;
    }                         
  }
}

#endif

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
 * \date Date of creation: 10/2014
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

#ifndef COB_SURFACE_POLICIES_HPP
#define COB_SURFACE_POLICIES_HPP

#include "cob_surface/geometry.h"

template<typename SurfaceT, typename Traits>
bool cob_surface::SweepLinePolicy<SurfaceT,Traits>::dataCompare(
  const DataT& a, const DataT& b, const StateT& state)
{
  ValueT y = state[1];
  if (a.op == Traits::FAKE)
  {
    ValueT bx = (y - projSpace(b.sf,b.v2)[1]) * b.xy_ratio + projSpace(b.sf,b.v2)[0];
    return a.xy_ratio < bx;
  }

  if (b.op == Traits::FAKE)
  {
    ValueT ax = (y - projSpace(a.sf,a.v2)[1]) * a.xy_ratio + projSpace(a.sf,a.v2)[0];
    return ax < b.xy_ratio;
  }

  // a & b are both starting edges at the same vertex
  if (projSpace(a.sf,a.v1) == projSpace(b.sf,b.v1))
    return projSpace(a.sf,a.v2)[0] < projSpace(b.sf,b.v2)[0];


  // intersection of y at a:
  ValueT ax = (y - projSpace(a.sf,a.v2)[1]) * a.xy_ratio + projSpace(a.sf,a.v2)[0];
  // intersection of y at b:
  ValueT bx = (y - projSpace(b.sf,b.v2)[1]) * b.xy_ratio + projSpace(b.sf,b.v2)[0];
  if (ax == bx)
  {
    std::cout << "dataCompare: swap point " << std::endl;
    return projSpace(a.sf,a.v2)[0] < projSpace(b.sf,b.v2)[0];
  }
  return ax < bx;
}

template<typename SurfaceT, typename Traits>
bool cob_surface::SweepLinePolicy<SurfaceT,Traits>::swapCheck(
  const DataT& a, const DataT& b, StateT& state)
{
  ValueT t,s;
  StateT a2 = projSpace(a.sf,a.v2);
  StateT b2 = projSpace(b.sf,b.v2);
  if (a2 == b2) return false; // same end point
  // assuming start points a1 and b1 are always different
  // else we would have a common insert event
  return Geometry::lineLineIntersection(
    projSpace(a.sf,a.v1), a2,
    projSpace(b.sf,b.v1), b2, state, t, s);
}

/*template<typename Traits>
static bool cob_surface::SweepLinePolicy<Traits>::dataCompare(
  const DataT& a, const DataT& b)
{} */

template<typename SurfaceT>
bool cob_surface::MergePolicy<SurfaceT>::updateVertex(
  const FaceHandle& face, const SurfaceT* sf_sensor,
  VertexHandle& vh_map, SurfaceT* sf_map)
{
  VertexHandle vh1, vh2, vh3;
  getFaceVertexHandles(sf_sensor, face, vh1, vh2, vh3);
  ScalarT u, v, w;
  Geometry::barycentric2d(
    projSpace(sf_map,vh_map), projSpace(sf_sensor,vh1),
    projSpace(sf_sensor,vh2), projSpace(sf_sensor,vh3), u, v, w);

  // the following update should be replaced by a kalman filter update:
  PointT p_sensor = u * mapSpace(sf_sensor,vh1) + 
    v * mapSpace(sf_sensor,vh2) + w * mapSpace(sf_sensor,vh3);
  mapSpace(sf_map,vh_map) += p_sensor;
  mapSpace(sf_map,vh_map) *= .5;
  return true;
}

template<typename SurfaceT>
bool cob_surface::MergePolicy<SurfaceT>::createVertex(
  const VertexHandle& vh_sensor, const SurfaceT* sf_sensor,
  const std::vector<FaceHandle>& triangles, VertexHandle& vh_map, SurfaceT* sf_map)
{
  std::vector<PointT> p_map(triangles.size());
  ScalarT min_dist = 100.;
  unsigned int min_idx = 0;

  for (unsigned int i=0; i<triangles.size(); ++i)
  {
    VertexHandle vh1, vh2, vh3;
    getFaceVertexHandles(sf_map, triangles[i], vh1, vh2, vh3);
    ScalarT u, v, w;
    Geometry::barycentric2d(
      projSpace(sf_sensor,vh_sensor), projSpace(sf_map,vh1),
      projSpace(sf_map,vh3),  projSpace(sf_map,vh3), u, v, w);
    p_map[i] = u * mapSpace(sf_map,vh1) + v * mapSpace(sf_map,vh2)
      + w * mapSpace(sf_map,vh3);
    ScalarT dist = (mapSpace(sf_sensor,vh_sensor) - p_map[i]).norm();
    if (dist < min_dist) { min_dist = dist; min_idx = i; }
  }
  // the following update should be replaced by a kalman filter update:
  bool updated;
  if (min_dist < 100.)
  {
    vh_map = sf_map->add_vertex( .5*(p_map[min_idx] + mapSpace(sf_sensor,vh_sensor)) );
    updated = true;
  }
  else
  {
    vh_map = sf_map->add_vertex( mapSpace(sf_sensor,vh_sensor) );
    updated = false;
  }
  projSpace(sf_map,vh_map) = projSpace(sf_sensor,vh_sensor);
  return updated;
}

template<typename SurfaceT>
typename SurfaceT::VertexHandle cob_surface::MergePolicy<SurfaceT>::createIntersection(
  const SurfaceT* sf1, const VertexHandle& vh11, const VertexHandle& vh12,
  const SurfaceT* sf2, const VertexHandle& vh21, const VertexHandle& vh22,
  const ProjPoint& p_proj, SurfaceT* sf_map)
{
  if (sf1 == sf2) return VertexHandle(); // return invalid vh
  // TODO: test whether new intersection computation is actually faster
  //   then computing the norm twice
  ScalarT s = (p_proj - projSpace(sf1,vh11)).norm();
  ScalarT t = (p_proj - projSpace(sf2,vh21)).norm();
  PointT p1 = mapSpace(sf1,vh11) * (1.-s) + mapSpace(sf1,vh12) * s;
  PointT p2 = mapSpace(sf2,vh21) * (1.-s) + mapSpace(sf2,vh22) * t;
  VertexHandle vh =  sf_map->add_vertex( .5*(p1+p2) );
  projSpace(sf_map,vh) = p_proj;
  return vh;
}

template<typename SurfaceT>
bool cob_surface::MergePolicy<SurfaceT>::vertexLeftRightOrder(
  const VertexHandle& vh1, const VertexHandle& vh2, const SurfaceT* sf_map)
{
  return projSpace(sf_map,vh1)[0] < projSpace(sf_map,vh2)[0];
  // check the y coordinate too?
}

template<typename SurfaceT>
void cob_surface::MergePolicy<SurfaceT>::createBoundingVertices(
  SurfaceT* sf_map, 
  const VertexHandle& vh_left,
  const VertexHandle& vh_right,
  const VertexHandle& vh_up,
  float alpha, VertexHandle& vh_left_bounding, VertexHandle& vh_right_bounding)
{
  ScalarT xmin = projSpace(sf_map,vh_left)[0];
  ScalarT xmax = projSpace(sf_map,vh_right)[0];
  ScalarT ymin = projSpace(sf_map,vh_up)[1];
  ScalarT d = alpha*(xmax - xmin);
  vh_left_bounding = sf_map->add_vertex(PointT(0,0,0));
  projSpace(sf_map,vh_left_bounding)[0] = xmin - d;
  projSpace(sf_map,vh_left_bounding)[1] = ymin;
  vh_right_bounding = sf_map->add_vertex(PointT(0,0,0));
  projSpace(sf_map,vh_right_bounding)[0] = xmax + d;
  projSpace(sf_map,vh_right_bounding)[1] = ymin;
}

#endif

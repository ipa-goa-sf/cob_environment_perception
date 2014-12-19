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
 * \date Date of creation: 11/2014
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

#ifndef COB_SURFACE_ADVANCING_FRONT_HPP
#define COB_SURFACE_ADVANCING_FRONT_HPP

template<typename SurfaceT, typename Policy>
template<typename IteratorT>
void SweepLine::AdvancingFront<SurfaceT,Policy>::finalize(
  const IteratorT& begin, const IteratorT& end)
{

}

template<typename SurfaceT, typename Policy>
void SweepLine::AdvancingFront<SurfaceT,Policy>::insertVertex(
  const VertexHandle& v)
{
  NodeIter r = bst_.upper_bound(v);
  NodeIter l = leftNeighbor(r);
  sf_->add_face(v, *l, *r);
  NodeIter x = bst_.insert(r,v);

  NodeIter tmp = rightNeighbor(r);
  bool fix_needed = true;
  while(tmp != bst_.end() && fix_needed)
  {
    fix_needed = fixNeighbor(r,tmp,x);
    r = tmp++;
  }

  fix_needed = true;
  tmp = l;
  while(tmp != bst_.begin() && fix_needed)
  {
    l = tmp--;
    fix_needed = fixNeighbor(l,x,tmp);
  }
}
/*
template<typename SurfaceT, typename Policy>
void SweepLine::AdvancingFront<SurfaceT,Policy>::insertEdge(
  const VertexHandle& v1, const VertexHandle& v2)
{
  insertVertex(v1);
  if (Policy::vertexCompare(v1,v2,sf_))
    fixEdgeRightMode(v1,v2);
  else
    fixEdgeLeftMode(v1,v2);
}

  // there are two traversal modes:
  // - triangle traversal: move along faces over their edges that
  //   get intersected by the constraining edge
  // - advancing front traversal: move along vertices of AF and
  //   check if it passes over the constraining edge

  // iterate all neighboring vertices vi of v1 and check if edge v2-v1
  // lies left or right of edge vi-v1 (change of sign (vi-v1)x(v2-v1) )

  // if no change of sign happened: proceed left/right on advancing front
  // check every vertex on AF if it lies above or below the edge.
  // if it lies below the edge -> intersection -> switch to triangle traversal


template<typename SurfaceT, typename Policy>
void SweepLine::AdvancingFront<SurfaceT,Policy>::fixEdgeRightMode(
  const VertexHandle& v1, const VertexHandle& v2)
{
  std::vector<VertexHandle> upper_vertices, lower_vertices;
  PointT p1 = sf_->point(v1);
  PointT e = sf_->point(v2) - p1;
  SurfaceT::VertexOHalfedgeIter he_it = sf_->voh_it(v1);
  SurfaceT::HalfedgeHandle heh;

  // first check vertices of all outgoing edges of v1
  // if and where a change of sign happens with respect to edge v1-v2
  // this determines the initial traverse mode and the halfedge handle to start with
  bool above = vertexPos(sf_->point(sf_->to_vertex(*he_it)), e, p1) < 0;
  bool face_traverse_mode = false;
  ++he_it;
  while(he_it.is_valid())
  {
    if ( above != (vertexPos(sf_->point(sf_->to_vertex(*he_it)), e, p1) < 0) )
    {
      if (above) heh = *he_it; // if we just changed to below
      else heh = *(--he_it); // if we just changed to above
      face_traverse_mode = true;
      break; // always take heh that is below -> heh on face
    }
    ++he_it;
  }

  // take some initial steps on surface depending on the traverse mode
  if (face_traverse_mode)
  {
    lower_vertices.push_back(sf_->to_vertex(heh));
    sf_->delete_face(sf_->face_handle(heh),false);

    heh = sf_->next_halfedge_handle(heh);
    upper_vertices.push_back(sf_->to_vertex(heh));
    heh = advance(heh);
  }
  else
  {
    heh = *(--he_it);
    if (!sf_->is_boundary(heh)) heh = advance(heh);
    lower_vertices.push_back(sf_->to_vertex(heh));
    heh = sf_->next_halfedge_handle(heh);
  }

  // iterate until end of edge is reached and remove all intersected triangles
  VertexHandle vh = sf_->to_vertex(heh);
  while (vh != v2)
  {
    // face traverse - the edge intersects existing triangles
    if (face_traverse_mode)
    {
      if (vertexPos(sf_->point(vh), e, p1) > 0) // is lower
      {
        lower_vertices.push_back(vh);
        heh = sf_->next_halfedge_handle(heh);
        if (sf_->is_boundary(vh)) face_traverse_mode = false;
      }
      else // is upper
      {
        upper_vertices.push_back(vh);
      }

      sf_->delete_face(sf_->face_handle(heh),false);
      heh = advance(heh);
      vh = sf_->to_vertex(heh);
    }
    // border traverse mode - the edge lies outside of the surface
    else
    {
      if (vertexPos(sf_->point(vh), e, p1) > 0) // is lower
      {
        lower_vertices.push_back(vh);
        heh = sf_->next_halfedge_handle(heh);
      }
      else
      {
        upper_vertices.push_back(vh);
        heh = advance(heh);
        face_traverse_mode = true;
      }
      vh = sf_->to_vertex(heh);
    }
  }

  // now retriangulate upper and lower vertices
}

template<typename SurfaceT, typename Policy>
void SweepLine::AdvancingFront<SurfaceT,Policy>::fixEdgeLeftMode(
  const VertexHandle& v1, const VertexHandle& v2)
{
  std::vector<VertexHandle> upper_vertices, lower_vertices;
  PointT p1 = sf_->point(v1);
  PointT e = sf_->point(v2) - p1;
  SurfaceT::VertexOHalfedgeIter he_it = sf_->voh_it(v1);
  SurfaceT::HalfedgeHandle heh;

  // first check vertices of all outgoing edges of v1
  // if and where a change of sign happens with respect to edge v1-v2
  // this determines the initial traverse mode and the halfedge handle to start with
  bool below = vertexPos(sf_->point(sf_->to_vertex(*he_it)), e, p1) < 0;
  bool face_traverse_mode = false;
  ++he_it;
  while(he_it.is_valid())
  {
    if ( below != (vertexPos(sf_->point(sf_->to_vertex(*he_it)), e, p1) < 0) )
    {
      if (below) heh = *he_it; // if we just changed to above
      else heh = *(--he_it); // if we just changed to below
      face_traverse_mode = true;
      break; // always take heh that is above -> heh on face
    }
    ++he_it;
  }

  // take some initial steps on surface depending on the traverse mode
  if (face_traverse_mode)
  {
    lower_vertices.push_back(sf_->to_vertex(heh));
    sf_->delete_face(sf_->face_handle(heh),false);

    heh = sf_->next_halfedge_handle(heh);
    upper_vertices.push_back(sf_->to_vertex(heh));
    heh = advance(heh);
  }
  else
  {
    heh = *(--he_it);
    if (!sf_->is_boundary(heh)) heh = advance(heh);
    lower_vertices.push_back(sf_->to_vertex(heh));
    heh = sf_->next_halfedge_handle(heh);
  }

  // iterate until end of edge is reached and remove all intersected triangles
  VertexHandle vh = sf_->to_vertex(heh);
  while (vh != v2)
  {
    // face traverse - the edge intersects existing triangles
    if (face_traverse_mode)
    {
      if (vertexPos(sf_->point(vh), e, p1) > 0) // is lower
      {
        lower_vertices.push_back(vh);
        heh = sf_->next_halfedge_handle(heh);
        if (sf_->is_boundary(vh)) face_traverse_mode = false;
      }
      else // is upper
      {
        upper_vertices.push_back(vh);
      }

      sf_->delete_face(sf_->face_handle(heh),false);
      heh = advance(heh);
      vh = sf_->to_vertex(heh);
    }
    // border traverse mode - the edge lies outside of the surface
    else
    {
      if (vertexPos(sf_->point(vh), e, p1) > 0) // is lower
      {
        lower_vertices.push_back(vh);
        heh = sf_->next_halfedge_handle(heh);
      }
      else
      {
        upper_vertices.push_back(vh);
        heh = advance(heh);
        face_traverse_mode = true;
      }
      vh = sf_->to_vertex(heh);
    }
  }

  // now retriangulate upper and lower vertices
}
*/

#endif

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
 * \date Date of creation: 09/2014
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

#ifndef COB_SURFACE_MERGE_HPP
#define COB_SURFACE_MERGE_HPP

#include "cob_surface/sweepline.h"
#include "cob_surface/traits.h"
#include "cob_surface/policies.h"

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::sensorIntoMap(
  const Mat4& tf_sensor,
  const Mat4& tf_map,
  const SurfaceT* sf_sensor,
  SurfaceT* sf_map)
{   
  // HOW TO DELETE FACES AND EDGES ???????????????
}

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::initialize(
  const SurfaceT* sf_sensor, SurfaceT* sf_map)
{
  VertexEventMap event_map_sensor;
  VertexEventMap event_map_map;

  sl_.reset(sf_sensors->n_edges() + sf_map->n_edges());

  SurfaceT::EdgeIter e_it;
  // create sweepline data elements from surface edges and determine
  // which side of the edge to process first.
  // Create events for every vertex and update these events
  // while we continue to create new data elements
  for (e_it=sf_sensor->edges_begin(); e_it!=sf_sensor->edges_end(); ++e_it)
    createEventForEdge(sf_sensor,*e_it, event_map_sensor);

  for (e_it=sf_map->edges_begin(); e_it!=sf_map->edges_end(); ++e_it)
    createEventForEdge(sf_map,*e_it), event_map_map;

  // add events to SweepLineProcess
  VertexEventMap::iterator map_it=event_map_sensor.begin()
  for (; map_it!=event_sensor.end(); ++map_it)
  {
    sl_.addEvent(map_it->second);
  }
   map_it=event_map_map.begin()
  for (; map_it!=event_map_map.end(); ++map_it)
  {
    sl_.addEvent(map_it->second);
  }
}

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::preprocess(SurfaceT* sf_map)
{
  /* This step iterates the sweepline to localize every vertex on a triangle
   * and use it to update the vertex position. 
   * It also determines the vertices that form new borders which is used
   * for the trianglulation step.
   */

  /* atm represents all currently active triangles and looks like this:
   * { []--(a)->[A]--(b)-->[A,B]--(c)->[B]--(d)->[] }
   * edge (a) transforms its previous bucket by activating triangle A
   * ...
   * edge (d) transforms its previous bucket by removing triangle B
   * atm assignes to every edge (DataId) the resulting bucket
   * of active triangles
   */
  ActiveTrianglesMap atm;
  sl_Event e;
  sl_DataId d_id;

  std::vector<VertexHandle> points_to_triangulate;
  std::vector<BorderEdge> border_edges;
  ActiveBorderVertices abv;

  std::vector<sl_DataId>::iterator it;
  while (sl_.nextEvent())
  {
    e = sl_.getCurrentEvent();

    // first: erase all buckets that resulted from recently removed edges
    for(it=e.to_remove.begin(); it!=e.to_remove.end(); ++it) atm.erase(*it);

    ActiveTrianglesBucket current_bucket;
    VertexHandle vh;

    // if event position was not at the beginning, we can use an existing
    // bucket, else use new empty bucket
    if (sl_.getLeftDataId(d_id)) current_bucket = atm[d_id];

    if (e.swap_event) // process intersecting edges
    {
      // First update ATM: remove the other bucket too and reinsert in swapped order
      atm.erase(e.to_insert[0]);

      transformActiveTrianglesBucket(sl_.getData(e.to_remove[0]), current_bucket);
      atm.insert(std::make_pair(e.to_remove[0], current_bucket));

      transformActiveTrianglesBucket(sl_.getData(e.to_insert[0]), current_bucket);
      atm.insert(std::make_pair(e.to_insert[0], current_bucket));

      // Next: create new intersection vertex and process the border vertex case
      sl_DataT& d1 = sl_.getData(e.to_remove[0]);
      sl_DataT& d2 = sl_.getData(e.to_insert[0]);
      vh = Policy::createIntersection(d1.sf, d1.v1, d1.v2, 
                                      d2.sf, d2.v1, d2.v2, e.state, sf_map);

      if (vh.is_valid() && !d1.f2.is_valid() && !d2.f2.is_valid())
      {
        // find and erase the data_id that has already be determined to 
        // be a border vertex and create a new edge
        // insert the current vh assigned with the other data_id back into ABV
        typename ActiveBorderVertices::iterator abv_it;
        if ( (abv_it=abv.find(e.to_remove[0])) != abv.end() )
        {
          border_edges.push_back(createBorderEdge(abv_it->second, vh, sf_map, d1));
          abv.erase(abv_it);
          abv.insert(e.to_insert[0],vh);
        }
        else if ( (abv_it=abv.find(e.to_insert[0])) != abv.end() )
        {
          border_edges.push_back(createBorderEdge(abv_it->second, vh, sf_map, d2));
          abv.erase(abv_it);
          abv.insert(e.to_remove[0],vh);
        }
        else
        {
          abv.insert(e.to_remove[0],vh);
          abv.insert(e.to_insert[0],vh);
        }
      }
    }
    else // process regular vertex
    {
      // First: use bucket to localize current vertex and update it
      sl_DataT* d;
      if (!e.to_insert.empty()) d = &sl_.getData(e.to_insert[0]); vh = d->v1;
      else                      d = &sl_.getData(e.to_remove[0]); vh = d->v2;

      bool updated;
      if(d->sf==sf_map) updated = updateVertex(current_bucket,vh,sf_map);
      else              updated = createVertex(current_bucket,vh,d->sf,vh,sf_map);

      // Next update ATM: just transform the current bucket and insert
      for(it=e.to_insert.begin(); it!=e.to_insert.end(); ++it)
      {
        transformActiveTrianglesBucket(sl_.getData(*it), current_bucket);
        atm.insert(std::make_pair(*it, current_bucket));
      }

      // Next: process the border vertex case
      if (!updated) // if the vertex was updated, its not on the border
      { // search for the two edges that lie on the border:
        for(it=e.to_remove.begin(); it!=e.to_remove.end(); ++it)
        { // if in remove list: create edge and erase from ABV
          if (sl_.getData(*it).f2.is_valid()) continue;
          typename ActiveBorderVertices::iterator abv_it = abv.find(*it);
          border_edges.push_back(createBorderEdge(abv_it->second, vh,
                                                  sf_map, sl_.getData(*it)));
          adv.erase(abv_it);
        }

        for(it=e.to_insert.begin(); it!=e.to_insert.end(); ++it)
        { // if in insert list: add current vertex to ABV
          if (sl_.getData(*it).f2.is_valid()) continue;
          abv.insert(*it, vh);
        }
      }
    }
    if (vh.is_valid()) points_to_triangulate.push_back(vh);
  }
}

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::triangulate(
  const std::vector<VertexHandle>& v_vh, SurfaceT* sf_map)
{
  /* This performs the last iteration over the data in order to
   * create a new triangulation of both original meshes.
   * For this all vertices are again processed in a sorted manner while
   * an advancing front (AF) maintains a sorted list of edges of the recently
   * created triangles. New vertices are being projected on this front in order
   * to find the edge that can be used to form a new triangle.
   */
  VertexHandle vh_left_most = v_vh[0];
  VertexHandle vh_right_most = v_vh[0];
  std::vector<VertexHandle>::iterator vh_it;
  for(vh_it = v_vh.begin()+1; vh_it!=v_vh.end(); ++vh_it)
  {
    if (Policy::vertexLeftRightOrder(*vh_it, vh_left_most, sf_map))
      vh_left_most = *vh_it;
    else if (Policy::vertexLeftRightOrder(vh_right_most, *vh_it, sf_map))
      vh_right_most = *vh_it;
  }
  Policy::createBoundingVertices(
    sf_map, vh_left_most, vh_right_most, .3f, vh_left_most, vh_right_most);

  SweepLine::AdvancingFront<> af(sf_map);
  af.initialize(vh_left_most, vh_right_most);
  for(vh_it = v_vh.begin(); vh_it!=v_vh.end(); ++vh_it)
    af.insertVertex(*vh_it);
  
}

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::createEventForEdge(
  const SurfaceT* sf, const EdgeHandle& eh, VertexEventMap& event_map)
{
  // determines the order of the edge vertices and checks the
  // orientation of the adjacent faces to create a new event
  // or update an existing one.
  FaceHandle f1, f2;
  HalfedgeHandle heh1 = sf->halfedge_handle(eh,0);
  HalfedgeHandle heh2 = sf->halfedge_handle(eh,1);
  VertexHandle v1 = sf->to_vertex(heh1);
  VertexHandle v2 = sf->to_vertex(heh2);
  sl_DataId data_id;
  SweepLineTraits::OperationType op;

  // first make sure that v1 comes before v2:
  if (SweepLinePolicy::stateCompare(projSpace(sf,v2), projSpace(sf,vh1)))
  {
    std::swap(v1,v2);
    std::swap(heh1,heh2);
  }

  // now check whether there are two faces and on which side
  // the third vertex of these faces lies
  if (sf->is_boundary(heh1))
  { // in case heh1 is boundary, we only have one face and v3
    HalfedgeHandle heh32 = sf->next_halfedge_handle(heh2);
    VertexHandle v32 = sf->to_vertex_handle(heh32);
    f1 = sf->face_handle(heh2);

    if (calcFaceOrientation(sf,v1,v2,v32) < 0)
      op = SweepLineTraits::DISABLE_SINGLE; // v32 on the left
    else
      op = SweepLineTraits::ENABLE_SINGLE; // v32 on the right
  }
  else if (sf->is_boundary(heh2))
  { // in case heh2 is boundary, we only have on face and v3
    HalfedgeHandle heh31 = sf->next_halfedge_handle(heh1);
    VertexHandle v31 = sf->to_vertex_handle(heh31);
    f1 = sf->face_handle(heh1);

    if (calcFaceOrientation(sf,v1,v2,v31) < 0)
      op = SweepLineTraits::DISABLE_SINGLE; // v31 on the left
    else
      op = SweepLineTraits::ENABLE_SINGLE; // v31 on the right
  }
  else
  { // else we have 2 faces, v31 and v32
    HalfedgeHandle heh31 = sf->next_halfedge_handle(heh1);
    HalfedgeHandle heh32 = sf->next_halfedge_handle(heh2);
    VertexHandle v31 = sf->to_vertex(heh31);
    VertexHandle v32 = sf->to_vertex(heh32);

    int state = int(calcFaceOrientation(sf,v1,v2,v31) < 0) << 1;
    state |= int(calcFaceOrientation(sf,v1,v2,v32) < 0);

    switch(state)
    {
    case(0b11):
    { // v31: (-), v32: (-) -> both left to the edge
      op = SweepLineTraits::DISABLE_DOUBLE; break;
    }
    case(0b00):
    { // v31: (+), v32: (+) -> both right to the edge
      op = SweepLineTraits::ENABLE_DOUBLE; break;
    }
    case(0b01):
    { // v31: (+), v32: (-) -> v32 left, v31 right
      std::swap(heh31,heh32); // fall through
    }
    case(0b10):
    { // v31: (-), v32: (+) -> v31 left, v32 right
      ob = SweepLineTraits::SWITCH; break;
    }
    }

    f1 = sf->face_handle(heh31);
    f2 = sf->face_handle(heh32);
  }

  data_id = sl_.addData(sl_DataT(sf, v1, v2, op, f1, f2));
  updateInsertEvent(sf, v1, data_id, event_map);
  updateRemoveEvent(sf, v2, data_id, event_map);
}

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::transformActiveTriangleBucket(
  const sl_DataT& data, ActiveTrianglesBucket& bucket)
{
  switch(data.op)
  {
  case(ENABLE_DOUBLE): bucket.push_back(std::make_pair(data.sf,data.f2));
  case(ENABLE_SINGLE): bucket.push_back(std::make_pair(data.sf,data.f1)); break;
  case(DISABLE_DOUBLE):
  {
    typename ActiveTriangleBucket::iterator it;
    for(it=bucket.begin();it!=bucket.end();++it)
    {
      if(it->second == data.f1 || it->second == data.f2) bucket.erase(it);
    }
    break;
  }
  case(DISABLE_SINGLE):
  {
    typename ActiveTriangleBucket::iterator it;
    for(it=bucket.begin();it!=bucket.end();++it)
    {
      if(it->second == data.f1) bucket.erase(it);
    }
    break;
  }
  case(SWITCH):
  {
    typename ActiveTriangleBucket::iterator it;
    for(it=bucket.begin();it!=bucket.end();++it)
    {
      if(it->second == data.f1) bucket.erase(it);
    }
    bucket.push_back(std::make_pair(data.sf,data.f2));
    break;
  }
  }
}

template<typename SurfaceT, typename Policy>
bool cob_surface::Merge<SurfaceT,Policy>::updateVertex(
  const ActiveTriangleBucket& bucket, VertexHandle& vh_map, SurfaceT* sf_map)
{
  // we search for the surface that belongs to the sensor (there is only one)
  typename ActiveTrianglesBucket::iterator b_it;
  for(b_it=bucket.begin(); b_it!=bucket.end(); ++b_it)
  {
    if (b_it->first != sf_map)
    { // update vh on sf_map with face b_it->second on b_it->first
      return Policy::updateVertex(b_it->second, b_it->first, vh_map, sf_map)
    }
  }
  return false;
}

template<typename SurfaceT, typename Policy>
bool cob_surface::Merge<SurfaceT,Policy>::createVertex(
  const ActiveTriangleBucket& bucket,
  const VertexHandle& vh_sensor, const SurfaceT* sf_sensor,
  VertexHandle& vh_map, SurfaceT* sf_map)
{
  // we search for all active triangles that belong to the map surface
  std::vector<FaceHandle> triangles;
  typename ActiveTrianglesBucket::iterator b_it;
  for(b_it=bucket.begin(); b_it!=bucket.end(); ++b_it)
  {
    if (b_it->first == sf_map) triangles.push_back(b_it->second);
  }

  // create a new vertex vh_map on sf_map that best matches the triangles
  // this should also delete all faces of sf_map that are in front of vh_sensor
  return Policy::createVertex(vh_sensor, sf_sensor, triangles, vh_map, sf_map);
}


#endif

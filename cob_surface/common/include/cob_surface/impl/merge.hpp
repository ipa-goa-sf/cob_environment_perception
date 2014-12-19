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


template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::initialize(
  const SurfaceT* sf_sensor, SurfaceT* sf_map)
{
  VertexEventMap event_map_sensor;
  VertexEventMap event_map_map;

  sl_.reset(sf_sensor->n_edges() + sf_map->n_edges());

  typename SurfaceT::EdgeIter e_it;
  // create sweepline data elements from surface edges and determine
  // which side of the edge to process first.
  // Create events for every vertex and update these events
  // while we continue to create new data elements
  for (e_it=sf_sensor->edges_begin(); e_it!=sf_sensor->edges_end(); ++e_it)
    createEventForEdge(sf_sensor,*e_it, event_map_sensor);

  for (e_it=sf_map->edges_begin(); e_it!=sf_map->edges_end(); ++e_it)
    createEventForEdge(sf_map,*e_it, event_map_map);

  // add events to SweepLineProcess
  typename VertexEventMap::iterator map_it=event_map_sensor.begin();
  for (; map_it!=event_map_sensor.end(); ++map_it)
  {
    sl_.addEvent(map_it->second);
  }
  map_it=event_map_map.begin();
  for (; map_it!=event_map_map.end(); ++map_it)
  {
    sl_.addEvent(map_it->second);
  }
}

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::preprocess(
  SurfaceT* sf_map,
  std::vector<VertexHandle>& points_to_triangulate,
  std::vector<BorderEdge>& border_edges)
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
  ActiveBorderVertices abv;
  points_to_triangulate.clear();
  border_edges.clear();
  std::vector<typename sl_Traits::StateT> states;
  std::vector<sl_DataId>::iterator it;
  while (sl_.nextEvent())
  {
    e = sl_.getCurrentEvent();
    states.push_back(e.state);

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
          abv.insert(std::make_pair(e.to_insert[0],vh));
        }
        else if ( (abv_it=abv.find(e.to_insert[0])) != abv.end() )
        {
          border_edges.push_back(createBorderEdge(abv_it->second, vh, sf_map, d2));
          abv.erase(abv_it);
          abv.insert(std::make_pair(e.to_remove[0],vh));
        }
        else
        {
          abv.insert(std::make_pair(e.to_remove[0],vh));
          abv.insert(std::make_pair(e.to_insert[0],vh));
        }
      }
    }
    else // process regular vertex
    {
      // First: use bucket to localize current vertex and update it
      sl_DataT* d;
      if (!e.to_insert.empty()) { d = &sl_.getData(e.to_insert[0]); vh = d->v1; }
      else                      { d = &sl_.getData(e.to_remove[0]); vh = d->v2; }

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
          if (abv_it == abv.end()) continue;
          border_edges.push_back(createBorderEdge(abv_it->second, vh,
                                                  sf_map, sl_.getData(*it)));
          abv.erase(abv_it);
        }

        for(it=e.to_insert.begin(); it!=e.to_insert.end(); ++it)
        { // if in insert list: add current vertex to ABV
          if (sl_.getData(*it).f2.is_valid()) continue;
          abv.insert(std::make_pair(*it, vh));
        }
      }
    }
    if (vh.is_valid()) points_to_triangulate.push_back(vh);
  }
  std::cout << "done" << std::endl;
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
  typename std::vector<VertexHandle>::const_iterator vh_it;
  for(vh_it = v_vh.begin()+1; vh_it!=v_vh.end(); ++vh_it)
  {
    if (Policy::vertexLeftRightOrder(*vh_it, vh_left_most, sf_map))
      vh_left_most = *vh_it;
    else if (Policy::vertexLeftRightOrder(vh_right_most, *vh_it, sf_map))
      vh_right_most = *vh_it;
  }
  Policy::createBoundingVertices(
    sf_map, vh_left_most, vh_right_most, v_vh[0], .3f, vh_left_most, vh_right_most);

  SweepLine::AdvancingFront<SurfaceT,Policy> af(sf_map);
  af.initialize(vh_left_most, vh_right_most);
  for(vh_it = v_vh.begin(); vh_it!=v_vh.end(); ++vh_it)
    af.insertVertex(*vh_it);

  sf_map->delete_vertex(vh_left_most,false);
  sf_map->delete_vertex(vh_right_most,false);
  sf_map->garbage_collection();
  // TODO: delete faces and bounding vertices
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
  VertexHandle v1 = sf->to_vertex_handle(heh1);
  VertexHandle v2 = sf->to_vertex_handle(heh2);
  sl_DataId data_id;
  
  typename sl_Traits::OperationType op;

  // first make sure that v1 comes before v2:
  if (sl_Policy::stateCompare(projSpace(sf,v2), projSpace(sf,v1)))
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

    if (areaFace(sf,v1,v2,v32) > 0)
      op = sl_Traits::DISABLE_SINGLE; // v32 on the left
    else
      op = sl_Traits::ENABLE_SINGLE; // v32 on the right
  }
  else if (sf->is_boundary(heh2))
  { // in case heh2 is boundary, we only have on face and v3
    HalfedgeHandle heh31 = sf->next_halfedge_handle(heh1);
    VertexHandle v31 = sf->to_vertex_handle(heh31);
    f1 = sf->face_handle(heh1);

    if (areaFace(sf,v1,v2,v31) > 0)
      op = sl_Traits::DISABLE_SINGLE; // v31 on the left
    else
      op = sl_Traits::ENABLE_SINGLE; // v31 on the right
  }
  else
  { // else we have 2 faces, v31 and v32
    HalfedgeHandle heh31 = sf->next_halfedge_handle(heh1);
    HalfedgeHandle heh32 = sf->next_halfedge_handle(heh2);
    VertexHandle v31 = sf->to_vertex_handle(heh31);
    VertexHandle v32 = sf->to_vertex_handle(heh32);

    int state = int(areaFace(sf,v1,v31,v2) > 0) << 1;
    state |= int(areaFace(sf,v1,v2,v32) > 0);

    switch(state)
    {
    case(0b01):
    { // v31: (+), v32: (-) -> cw/ccw -> both left to the edge
      op = sl_Traits::DISABLE_DOUBLE; break;
    }
    case(0b10):
    { // v31: (-), v32: (+) -> ccw/cw -> both right to the edge
      op = sl_Traits::ENABLE_DOUBLE; break;
    }
    case(0b00):
    { // v31: (+), v32: (+) -> both cw -> v32 right, v31 left
      // this is actually the case when we look from behind (needs some validation)
      std::swap(heh31,heh32); // fall through
    }
    case(0b11):
    { // v31: (-), v32: (-) -> both ccw -> v32 left, v31 right
      op = sl_Traits::SWITCH; break;
    }
    }

    f1 = sf->face_handle(heh32); // turn off T(v1,v2,v32)
    f2 = sf->face_handle(heh31); // turn on T(v1,v31,v2)
  }

  data_id = sl_.addData(sl_DataT(sf, v1, v2, op, f1, f2));
  updateInsertEvent(sf, v1, data_id, event_map);
  updateRemoveEvent(sf, v2, data_id, event_map);
}

template<typename SurfaceT, typename Policy>
void cob_surface::Merge<SurfaceT,Policy>::transformActiveTrianglesBucket(
  const sl_DataT& data, ActiveTrianglesBucket& bucket)
{
  switch(data.op)
  {
  case(sl_Traits::ENABLE_DOUBLE):
    bucket.push_back(std::make_pair(data.sf,data.f2));
  case(sl_Traits::ENABLE_SINGLE):
    bucket.push_back(std::make_pair(data.sf,data.f1)); break;
  case(sl_Traits::DISABLE_DOUBLE):
  {
    typename ActiveTrianglesBucket::iterator it = bucket.begin();
    while(it!=bucket.end())
    {
      if(it->second == data.f1 || it->second == data.f2) bucket.erase(it++);
      else ++it;                                              
    }
    break;
  }
  case(sl_Traits::DISABLE_SINGLE):
  {
    typename ActiveTrianglesBucket::iterator it = bucket.begin();
    while(it!=bucket.end())
    {
      if(it->second == data.f1) bucket.erase(it++);
      else ++it;
    }
    break;
  }
  case(sl_Traits::SWITCH):
  {
    typename ActiveTrianglesBucket::iterator it = bucket.begin();
    while(it!=bucket.end())
    {
      if(it->second == data.f1) bucket.erase(it++);
      else ++it;
    }
    bucket.push_back(std::make_pair(data.sf,data.f2));
    break;
  }
  case(sl_Traits::FAKE):
  {
    assert(data.op != sl_Traits::FAKE);
  }
  }
}

template<typename SurfaceT, typename Policy>
bool cob_surface::Merge<SurfaceT,Policy>::updateVertex(
  const ActiveTrianglesBucket& bucket, VertexHandle& vh_map, SurfaceT* sf_map)
{
  // we search for the surface that belongs to the sensor (there is only one)
  typename ActiveTrianglesBucket::const_iterator b_it;
  for(b_it=bucket.begin(); b_it!=bucket.end(); ++b_it)
  {
    if (b_it->first != sf_map)
    { // update vh on sf_map with face b_it->second on b_it->first
      return Policy::updateVertex(b_it->second, b_it->first, vh_map, sf_map);
    }
  }
  return false;
}

template<typename SurfaceT, typename Policy>
bool cob_surface::Merge<SurfaceT,Policy>::createVertex(
  const ActiveTrianglesBucket& bucket,
  const VertexHandle& vh_sensor, const SurfaceT* sf_sensor,
  VertexHandle& vh_map, SurfaceT* sf_map)
{
  // we search for all active triangles that belong to the map surface
  std::vector<FaceHandle> triangles;
  typename ActiveTrianglesBucket::const_iterator b_it;
  for(b_it=bucket.begin(); b_it!=bucket.end(); ++b_it)
  {
    if (b_it->first == sf_map) triangles.push_back(b_it->second);
  }

  // create a new vertex vh_map on sf_map that best matches the triangles
  // this should also delete all faces of sf_map that are in front of vh_sensor
  return Policy::createVertex(vh_sensor, sf_sensor, triangles, vh_map, sf_map);
}


#endif

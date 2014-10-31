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

template<typename Traits, typename Policy>
void cob_surface::Merge<Traits,Policy>::sensorIntoMap(
  const Mat4& tf_sensor,
  const Mat4& tf_map,
  const SurfaceT& sf_sensor,
  SurfaceT& sf_map)
{   
    // HOW TO DELETE FACES AND EDGES ???????????????
}

template<typename Traits, typename Policy>
void cob_surface::Merge<Traits,Policy>::initialize(
  const SurfaceT& sf_sensor, SurfaceT& sf_map)
{
  typedef typename SweepLineTraits<SurfaceT>::DataT DataT;
  VertexEventMap event_map;

  sl_.reset(sf_sensors.n_edges() + sf_map.n_edges());

  SurfaceT::EdgeIter e_it;
  // create sweepline data elements from surface edges and determine
  // which side of the edge to process first.
  // Create events for every vertex and update these events
  // while we continue to create new data elements
  for (e_it=sf_sensor.edges_begin(); e_it!=sf_sensor.edges_end(); ++e_it)
  {
    SurfaceT::VertexHandle v1 = getVertexHandle(sf_sensor,*e_it,0);
    SurfaceT::VertexHandle v2 = getVertexHandle(sf_sensor,*e_it,1);
    if (SweepLinePolicy::stateCompare(sf_sensor.point(v1), sf_sensor.point(v2)))
    {
      SweepLine::DataId data_id = sl_.addData({&sf_sensor,v1,v2});
      updateInsertEvent(sf_sensor, v1, data_id, event_map);
      updateRemoveEvent(sf_sensor, v2, data_id, event_map);
    }
    else
    {
      SweepLine::DataId data_id = sl_.addData({&sf_sensor,v2,v1});
      updateInsertEvent(sf_sensor, v2, data_id, event_map);
      updateRemoveEvent(sf_sensor, v1, data_id, event_map);
    }
  }

  // do the same for map surface
  for (e_it=sf_map.edges_begin(); e_it!=sf_map.edges_end(); ++e_it)
  {
    SurfaceT::VertexHandle v1 = getVertexHandle(sf_map,*e_it,0);
    SurfaceT::VertexHandle v2 = getVertexHandle(sf_map,*e_it,1);
    if (SweepLinePolicy::stateCompare(sf_map.point(v1), sf_map.point(v2)))
    {
      SweepLine::DataId data_id = sl_.addData({&sf_map,v1,v2});
      updateInsertEvent(sf_map, v1, data_id, event_map);
      updateRemoveEvent(sf_map, v2, data_id, event_map);
    }
    else
    {
      SweepLine::DataId data_id = sl_.addData({&sf_map,v2,v1});
      updateInsertEvent(sf_map, v2, data_id, event_map);
      updateRemoveEvent(sf_map, v1, data_id, event_map);
    }
  }

  // add events to SweepLineProcess
  VertexEventMap::iterator map_it=event_map.begin()
  for (; map_it!=event_map.end(); ++map_it)
  {
    // todo: sort data list of events
    sl_.addEvent(map_it->second);
  }
}


template<typename Traits, typename Policy>
void cob_surface::Merge<Traits,Policy>::preprocess()
{
  /* at_map represents all currently active triangles and looks like this:
   * { []--(a)->[A]--(b)-->[A,B]--(c)->[B]--(d)->[] }
   * edge a transforms its previous bucket by activating triangle A
   * ...
   * edge d transforms its previous bucket by removing triangle B
   * at_map assignes to every edge (DataId) the resulting bucket
   * of active triangles
   */
  ActiveTrianglesMap at_map;
  SweepLineProcess::Event e;
  SweepLine::DataId d_id;

  std::vector<DataId>::iterator it;
  while (sl_.nextEvent())
  {
    e = sl_.getCurrentEvent();

    // first: erase all buckets that resulted from recently removed edges
    for(it=e.to_remove.begin(); it!=e.to_remove.end(); ++it) at_map.erase(*it);

    ActiveTrianglesBucket current_bucket;

    // if event position was not at the beginning, we can use an existing
    // bucket, else use new empty bucket
    if (sl_.getLeftData(d_id))
    {
      current_bucket = at_map[d_id];
    }

    // we can now use this bucket to localize the current event on a triangle
    if (current_bucket.empty()) // we have a free vertex
    {
    }
    else if (!e.to_insert.empty())
    {
      DataT& data = sl_.getData(e.to_insert[0]);
      updateVertex(current_bucket, data);
    }
    else
    {
      DataT& data = sl_.getData(e.to_remove[0]);
      updateVertex(current_bucket, data);
    }

    if (e.swap_event)
    {
      // remove both buckets and reinsert in swapped order
      at_map.erase(e.to_insert[0]);

      transformActiveTrianglesBucket(sl_.getData(e.to_remove[0]), current_bucket);
      at_map.insert(std::make_pair(e.to_remove[0], current_bucket));

      transformActiveTrianglesBucket(sl_.getData(e.to_insert[0]), current_bucket);
      at_map.insert(std::make_pair(e.to_insert[0], current_bucket));
    }
    else
    {
      for(it=e.to_insert.begin(); it!=e.to_insert.end(); ++it)
      {
        transformActiveTrianglesBucket(sl_.getData(*it), current_bucket);
        at_map.insert(std::make_pair(*it, current_bucket));
      }
    }
  }
}

template<typename Traits, typename Policy>
void cob_surface::Merge<Traits,Policy>::transformActiveTriangleBucket(
  const DataT& data, ActiveTrianglesBucket& bucket)
{
  switch(data.op)
  {
  case(ENABLE_DOUBLE): bucket.push_back(std::make_pair(data.sf,data.f2));
  case(ENABLE_SINGLE): bucket.push_back(std::make_pair(data.sf,data.f1)); break;
  case(DISABLE_DOUBLE):
  {
    for(ActiveTriangleBucket::iterator it=bucket.begin();it!=bucket.end();++it)
    {
      if(it->second == data.f1 || it->second == data.f2) bucket.erase(it);
    }
    break;
  }
  case(DISABLE_SINGLE):
  {
    for(ActiveTriangleBucket::iterator it=bucket.begin();it!=bucket.end();++it)
    {
      if(it->second == data.f1) bucket.erase(it);
    }
    break;
  }
  case(SWITCH):
  {
    for(ActiveTriangleBucket::iterator it=bucket.begin();it!=bucket.end();++it)
    {
      if(it->second == data.f1) bucket.erase(it);
    }
    bucket.push_back(std::make_pair(data.sf,data.f2));
    break;
  }
  }
}

template<typename Traits, typename Policy>
void cob_surface::Merge<Traits,Policy>::updateVertex(
  const ActiveTriangleBucket& bucket, DataT& data)
{
  // here we search for all active triangles that belong to a different
  // surface as the vertex of the current event
  std::vector<SurfaceT::FaceHandle> triangles;
  SurfaceT* sf_other;
  typename ActiveTrianglesBucket::iterator b_it;
  for(b_it=bucket.begin(); b_it!=bucket.end(); ++b_it)
  {
    if (b_it->first != data.sf)
    {
      triangles.push_back(b_it->second);
      sf_other = b_it->first;
    }
  }
  // now we can update the current vertex using the best match of triangles
  Policy::updateVertex(data.v1, data.sf, triangles, sf_other);  
}


#endif

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

#ifndef COB_SURFACE_MERGE_H
#define COB_SURFACE_MERGE_H

#include <unordered_map>

#include <cob_surface/typedefs.h>

namespace cob_surface
{
  template<typename SurfaceT, typename Policy=cob_surface::MergePolicy<SurfaceT> >
  class Merge
  {
  public:
    typedef typename SurfaceT::VertexHandle VertexHandle;
    typedef typename SurfaceT::FaceHandle FaceHandle;
    typedef typename SurfaceT::EdgeHandle EdgeHandle;
    typedef typename SurfaceT::HalfedgeHandle HalfedgeHandle;
    typedef typename SurfaceT::Point Point;

    // some abbreviation for SweepLine specific types:
    typedef SweepLineTraits<SurfaceT> sl_Traits;
    typedef SweepLinePolicy sl_Policy;
    typedef SweepLine::Event<sl_Traits> sl_Event;
    typedef SweepLine::DataId sl_DataId;
    typedef typename sl_Traits::DataT sl_DataT;
    typedef SweepLine::SweepLineProcess<sl_Traits,sl_Policy> sl_Process;


    typedef std::unordered_map<VertexHandle, sl_Event> VertexEventMap;
    typedef std::list<std::pair<SurfaceT*,FaceHandle> > ActiveTrianglesBucket;
    typedef std::unordered_map<sl_DataId,ActiveTrianglesBucket> ActiveTrianglesMap;
    typedef std::unordered_map<sl_DataId,VertexHandle> ActiveBorderVertices;

    struct BorderEdge
    {
      VertexHandle v1; //< upper vertex
      VertexHandle v2; //< lower vertex
      bool openning; //< whether the edge is openning or closing the surface
    };

  public:
    /*
      merges sensor surface into map surface
    */
    void sensorIntoMap(
      const Mat4& tf_sensor,
      const Mat4& tf_map,
      const SurfaceT* sf_sensor,
      SurfaceT* sf_map);

    void initialize(const SurfaceT* sf_sensor, SurfaceT* sf_map);

    void preprocess(SurfaceT* sf_map);

    void triangulate(const std::vector<VertexHandle>& v_vh, SurfaceT* sf_map);


  private:
    void updateInsertEvent(const SurfaceT* sf, const VertexHandle& vh,
                           const sl_DataId& d_id, VertexEventMap& map)
    {
      if (map.count(vh)) map[vh].to_insert.push_back(d_id);
      else
      {
        sl_Event ev = {projSpace(sf,vh), {d_id}, {}, false};
        map.insert(std::make_pair(vh, ev));
      }
    }

    void updateRemoveEvent(const SurfaceT* sf, const VertexHandle& vh,
                           const sl_DataId& d_id, VertexEventMap& map)
    {
      if (map.count(vh)) map[vh].to_remove.push_back(d_id);
      else
      {
        sl_Event ev = {projSpace(sf,vh), {}, {d_id}, false};
        map.insert(std::make_pair(vh, ev));
      }
    }

    void createEventForEdge(const SurfaceT* sf, const EdgeHandle& eh,
                            VertexEventMap& event_map);

    void transformActiveTrianglesBucket(const sl_DataT& data,
                                        ActiveTrianglesBucket& bucket);

    bool updateVertex(const ActiveTriangleBucket& bucket,
                      VertexHandle& vh_map,
                      SurfaceT* sf_map);

    bool createVertex(const ActiveTriangleBucket& bucket,
                      const VertexHandle& vh_sensor,
                      const SurfaceT* sf_sensor,
                      VertexHandle& vh_map,
                      SurfaceT* sf_map);

    /** 
     * check if edge between vh1 and vh2 is an opening or closing edge
     * 
     * @param op - the triangle operation performed by this edge
     * @param vh1 - the upper vertex
     * @param vh2 - the lower vertex
     * @param sf_map - point to surface both vh belong to
     * 
     * @return true if edge is opening, false if closing
     */
    inline bool isOpeningEdge(const SweepLineTraits::OperationType& op,
                              const VertexHandle& vh1,
                              const VertexHandle& vh2,
                              const SurfaceT* sf_map)
    {
      /* The following logic applies:
       *  (EN: enable single, R: vh1 is to the right of vh2)
       *  out  |  EN   R
       * ---------------
       *   1   |  0   0            ==>  !(EN xor R)
       *   0   |  0   1         which is (EN == R)
       *   0   |  1   0
       *   1   |  1   1
       */
      return ( (op == SweepLineTraits::ENABLE_SINGLE)
               == Policy::vertexLeftRightOrder(vh2,vh1,sf_map) )
    }

    inline BorderEdge createBorderEdge(const VertexHandle& vh1,
                                       const VertexHandle& vh2,
                                       const SurfaceT* sf_map,
                                       const sl_DataT& data)
    {
      return {vh1, vh2, isOpeningEdge(data.op, vh1, vh2, sf_map) };
    }
    

  private:
    sl_Process sl_;
    

  };
}


#endif

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

#ifndef COB_SURFACE_POLICIES_H
#define COB_SURFACE_POLICIES_H

/* This header defines all policies necessary for using
 * the surface class with several algorithms
 */

#include "cob_surface/surface.h"

namespace cob_surface
{
  // define policy for surface conversion
  struct ConversionPolicy
  {
    typedef Surface::VertexHandle VHandle;
    typedef Surface::FaceHandle FHandle;

    template<typename PointT>
    static inline VHandle addVertex(
      const PointT& p, unsigned int idx, Surface& sf)
    {
      return sf.add_vertex(Surface::Point(p.getVector3fMap()));
    }

    static inline FHandle addFace(
      const VHandle& v1, const VHandle& v2, const VHandle& v3, Surface& sf)
    {
      return sf.add_face(v1,v2,v3);
    }
  };

  // define policy for using SweepLineProcess on surface class
  template<Traits = SweepLineTraits>
  struct SweepLinePolicy
  {
    typedef typename Traits::ValueT ValueT;
    typedef typename Traits::DataT DataT;
    typedef typename Traits::StateT StateT;
    typedef typename Traits::SurfaceT SurfaceT;
    /** 
     * Determines the order between two states a and b
     * 
     * @param a - State a
     * @param b - State b
     * 
     * @return returns true if a comes before b
     */
    inline static bool stateCompare(const StateT& a, const StateT& b)
    {
      if (a[1] != b[1]) return a[1] < b[1];
      return a[0] < b[0];
    }

    /** 
     * Determines the order between data a and b at a given state
     * 
     * @param a - Data a
     * @param b - Data b
     * @param state - the current state at which to evaluate
     * 
     * @return returns true if a comes before b
     */
    static bool dataCompare(const DataT& a, const DataT& b, const StateT& state);

    /** 
     * Determines the order between data a and b if both share
     * one common vertex (v1 or v2)
     * 
     * @param a - Data a
     * @param b - Data b
     * 
     * @return - returns true if a commes before b
     */
    //static bool dataCompare(const DataT& a, const DataT& b);

    /** 
     * Creates a dummy data set that allows to locate a state within bst
     * This data will be used by dataCompare()
     * 
     * @param state - the state that should be located
     * @param out - the created dummy data
     */
    inline static void dataForLocalization(const StateT& state, DataT& out)
    {
      out.op = SweepLineTraits::FAKE;
      out.xy_ratio = state[0];
    }

    /** 
     * tests for 2D line line intersection
     * 
     * @param a 
     * @param b 
     * @param state 
     * 
     * @return 
     */
    static bool swapCheck(const DataT& a, const DataT& b, StateT& state);
  };

  // define policy for merging two surfaces
  template<typename SurfaceT>
  struct MergePolicy
  {
    typedef typename SurfaceT::Scalar ScalarT;
    typedef typename SurfaceT::Point PointT;
    typedef typename SurfaceT::VertexHandle VertexHandle;
    typedef typename SurfaceT::FaceHandle FaceHandle;    

    static bool updateVertex(
      const FaceHandle& face,
      const SurfaceT* sf_sensor,
      VertexHandle& vh_map,
      SurfaceT* sf_map);

    static bool createVertex(
      const VertexHandle& vh_sensor,
      const SurfaceT* sf_sensor,
      const std::vector<FaceHandle>& triangles,
      VertexHandle& vh_map,
      SurfaceT* sf_map);

    static VertexHandle createIntersection(
      const SurfaceT* sf1,
      const VertexHandle& vh11,
      const VertexHandle& vh12,
      const SurfaceT* sf2,
      const VertexHandle& vh21,
      const VertexHandle& vh22,
      const StateT& p_proj,
      SurfaceT* sf_map);

    /** 
     * Checks whether vh1 is left to vh2
     * 
     * @param vh1 
     * @param vh2 
     * @param sf_map 
     * 
     * @return return true if vh1 is left to vh2
     */
    static bool vertexLeftRightOrder(
      const VertexHandle& vh1,
      const VertexHandle& vh2,
      const SurfaceT* sf_map);

    static void createBoundingVertices(
      SurfaceT* sf_map,      
      const VertexHandle& vh_left,
      const VertexHandle& vh_right,
      float alpha,
      VertexHandle& vh_left_bounding,
      VertexHandle& vh_right_bounding);
  };
}

#include "cob_surface/impl/policies.hpp"

#endif

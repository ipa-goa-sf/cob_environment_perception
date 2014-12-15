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

#ifndef COB_SURFACE_TRAITS_H
#define COB_SURFACE_TRAITS_H

#include "cob_surface/surface.h"

namespace cob_surface
{
  template<typename SurfaceT>
  struct SweepLineTraits
  {
    typedef typename SurfaceT::Scalar ValueT;
    typedef typename SurfaceT::ProjPoint StateT;
    typedef typename SurfaceT::VertexHandle VertexHandle;
    typedef typename SurfaceT::FaceHandle FaceHandle;
    typedef typename SurfaceT::HalfedgeHandle HalfedgeHandle;


    /* Definition of operations for the active triangle list.
     *
     * Every surface edge can have either one or two adjacent triangles.
     * When the active edge list of the current sweepline changes, we
     * also adjust the list of currently active triangles.
     * When the sweepline intersects an edge, a triangle can either be
     * turned active or inactive.
     */
    enum OperationType
    {
      ENABLE_SINGLE = 1, // turn on f1
      ENABLE_DOUBLE = 2, // turn on f1 and f2
      DISABLE_SINGLE = 3, // turn off f1
      DISABLE_DOUBLE = 4, // turn off f1 and f2
      SWITCH = 5, // turn off f1, turn on f2
      FAKE = 6 // fake data for localization
    };

    struct DataT
    {
      DataT() {}
      DataT(const SurfaceT* sf_, const VertexHandle& v1_, const VertexHandle& v2_,
            OperationType op_, const FaceHandle& f1_, const FaceHandle& f2_)
        : sf(sf_), v1(v1_), v2(v2_), op(op_), f1(f1_), f2(f2_)
      {
        StateT d = projSpace(sf_,v1_) - projSpace(sf_,v2_);
        xy_ratio = - d[0] / d[1];
      }

      const SurfaceT* sf;
      VertexHandle v1; //< upper vertex
      VertexHandle v2; //< lower vertex
      OperationType op;
      FaceHandle f1;
      FaceHandle f2;
      ValueT xy_ratio;
    };
  };
}

#endif

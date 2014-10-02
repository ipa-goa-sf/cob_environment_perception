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

#include <queue>
#include <set>

#include <cob_surface/typedefs.h>

namespace cob_surface
{
  // struct for elements in event schedule
  struct Event
  {
    Event()
    {
    }

    inline bool isIntersection()
    {
    }

    float x; // coord in sweep space
    float y; // coord in sweep space
    // list of starting edges, ref to surface
    // list of ending edges, ref to surface
    // pair of intersecting edges, ref to surface

  };

  // struct for elements on sweepline
  struct SweepObject
  {
    SweepObject()
    {
    }

  };

  inline const bool operator< (const Event& lhs, const Event& rhs)
  { return lhs.y < rhs.y; }
  inline const bool operator> (const Event& lhs, const Event& rhs)
  { return operator< (rhs, lhs); }
  inline const bool operator<= (const Event& lhs, const Event& rhs)
  { return !operator> (lhs, rhs); }
  inline const bool operator>= (const Event& lhs, const Event& rhs)
  { return !operator< (lhs, rhs); }


  template<typename SurfaceT>
  class Merge
  {
  public:
    /*
      merges sensor surface into map surface
    */
    static void sensorIntoMap(
      const Mat4& tf_sensor,
      const Mat4& tf_map,
      const SurfaceT& sf_sensor,
      SurfaceT& sf_map)
  };

}


#endif

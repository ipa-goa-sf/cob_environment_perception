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


#ifndef COB_SURFACE_CONVERSION_HPP
#define COB_SURFACE_CONVERSION_HPP

#include <vector>


template<typename SensorT>
template<typename PointT, typename MeshT, typename Policy>
void cob_surface::Conversion<SensorT>::pointCloud2Mesh(
  const typename pcl::PointCloud<PointT>::ConstPtr& pc, MeshT& mesh)
{
  typedef typename MeshT::VertexHandle VHandle;

  // maintains horizontal status of last line
  std::vector<bool> h1(pc->width-1, false);
  bool v1; // maintains last vertical status
  
  // maintains vertex handles of last line
  std::vector<VHandle> vhandle(pc->width);
  VHandle vhandle21; // maintains handles of current line

  unsigned int idx = 1;
  for (; idx < pc->width; ++idx) // process first row
  {
    h1[idx-1] = isNeighbor(pc->points[idx].getVector3fMap(),
                           pc->points[idx-1].getVector3fMap());
  }

  while(idx < pc->size()) // iterate rows
  {
    v1 = isNeighbor(pc->points[idx].getVector3fMap(),
                    pc->points[idx - pc->width].getVector3fMap());
    ++idx;

    for(unsigned int c = 1; c<pc->width; ++c) // iterate cols
    {
      /*
       *  +--+    p11 - h1 - p12
       *  |  |    v1          v2
       *  +--+    p21 - h2 - p22
       */

      unsigned int i22 = idx;
      unsigned int i21 = idx-1;
      unsigned int i12 = idx - pc->width;
      unsigned int i11 = idx-1 - pc->width;

      const PointT* p22 = &(pc->points[i22]);
      const PointT* p21 = &(pc->points[i21]);
      const PointT* p12 = &(pc->points[i12]);
      const PointT* p11 = &(pc->points[i11]);

      VHandle vhandle22;

      bool h2 = isNeighbor(p21->getVector3fMap(), p22->getVector3fMap());
      bool v2 = isNeighbor(p12->getVector3fMap(), p22->getVector3fMap());
      unsigned char status = (v2<<3) | (h2<<2) | (v1<<1) | h1[c-1];

      switch(status)
      {
      case 0b1111: case 0b1110: case 0b1101: case 0b1011: case 0b0111:
      { // three or four valid edges -> create both triangles
        if (!vhandle[c-1].is_valid()) vhandle[c-1] = Policy::addVertex(*p11, i11, mesh);
        if (!vhandle[c].is_valid()) vhandle[c] = Policy::addVertex(*p12, i12, mesh);
        if (!vhandle21.is_valid()) vhandle21 = Policy::addVertex(*p21, i21, mesh);
        vhandle22 = Policy::addVertex(*p22, i22, mesh);

        if ((p21->z - p12->z) > (p22->z - p11->z))
        {
          Policy::addFace(vhandle22, vhandle[c], vhandle[c-1], mesh);
          Policy::addFace(vhandle22, vhandle[c-1], vhandle21, mesh);
        }
        else
        {
          Policy::addFace(vhandle21, vhandle[c], vhandle[c-1], mesh);
          Policy::addFace(vhandle21, vhandle22, vhandle[c], mesh);
        }
        break;
      }
      case 0b0011: // upper left triangle. CCW: p21 - p12 - p11
      {
        if (!vhandle21.is_valid()) vhandle21 = Policy::addVertex(*p21, i21, mesh);
        if (!vhandle[c].is_valid()) vhandle[c] = Policy::addVertex(*p12, i12, mesh);
        if (!vhandle[c-1].is_valid()) vhandle[c-1] = Policy::addVertex(*p11, i11, mesh);

        Policy::addFace(vhandle21, vhandle[c], vhandle[c-1], mesh);
        break;
      }
      case 0b1100: // lower right triangle. ccw: p21 - p22 - p12
      {
        if (!vhandle21.is_valid()) vhandle21 = Policy::addVertex(*p21, i21, mesh);
        vhandle22 = Policy::addVertex(*p22, i22, mesh);
        if (!vhandle[c].is_valid()) vhandle[c] = Policy::addVertex(*p12, i12, mesh);

        Policy::addFace(vhandle21, vhandle22, vhandle[c], mesh);
        break;
      }
      case 0b1001: // upper right triangle. ccw: p22 - p12 - p11
      {
        vhandle22 = Policy::addVertex(*p22, i22, mesh);
        if (!vhandle[c].is_valid()) vhandle[c] = Policy::addVertex(*p12, i12, mesh);
        if (!vhandle[c-1].is_valid()) vhandle[c-1] = Policy::addVertex(*p11, i11, mesh);

        Policy::addFace(vhandle22, vhandle[c], vhandle[c-1], mesh);
        break;
      }
      case 0b0110: // lower left triangle. ccw: p21 - p22 - p11
      {
        if (!vhandle21.is_valid()) vhandle21 = Policy::addVertex(*p21, i21, mesh);
        vhandle22 = Policy::addVertex(*p22, i22, mesh);
        if (!vhandle[c-1].is_valid()) vhandle[c-1] = Policy::addVertex(*p11, i11, mesh);

        Policy::addFace(vhandle22, vhandle[c-1], vhandle21, mesh);
        break;
      }
      default:
        break;
      }

      h1[c-1] = h2;
      v1 = v2;
      vhandle[c-1] = vhandle21;
      vhandle21 = vhandle22;

      ++idx;
    }
  }
}

#endif

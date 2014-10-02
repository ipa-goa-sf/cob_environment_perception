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

#include <set>
#include <iostream>
#include <Eigen/Core>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>



struct SurfaceTraits : public OpenMesh::DefaultTraits
{
  typedef Eigen::Vector3f Point;
};

namespace OpenMesh
{
  template<>
  struct vector_traits<Eigen::Vector3f>
  {
    typedef Eigen::Vector3f vector_type;
    typedef float value_type;
    static size_t size() { return 3; }
    static const size_t size_ = 3;
  };
}

typedef OpenMesh::TriMesh_ArrayKernelT<SurfaceTraits> Surface;

int main(int argc, char **argv)
{
  Surface sf;
  Surface::VertexHandle vh[4];
  vh[0] = sf.add_vertex(Surface::Point(0.,0.,0.));
  vh[1] = sf.add_vertex(Surface::Point(1.,0.,0.));
  vh[2] = sf.add_vertex(Surface::Point(0.,1.,0.));
  vh[3] = sf.add_vertex(Surface::Point(1.,1.,1.));

  std::vector<Surface::VertexHandle> fvh;
  fvh.push_back(vh[2]);
  fvh.push_back(vh[1]);
  fvh.push_back(vh[0]);
  sf.add_face(fvh);

  fvh.clear();
  fvh.push_back(vh[2]);
  fvh.push_back(vh[3]);
  fvh.push_back(vh[1]);
  sf.add_face(fvh);

  fvh.clear();
  fvh.push_back(vh[1]);
  fvh.push_back(vh[3]);
  fvh.push_back(vh[0]);
  sf.add_face(fvh);

  fvh.clear();
  fvh.push_back(vh[3]);
  fvh.push_back(vh[2]);
  fvh.push_back(vh[0]);
  sf.add_face(fvh);

  //OpenMesh::IO::write_mesh(sf, "/home/goa-sf/Desktop/test.ply");
  std::cout << "Hallo" << std::endl;
  return 0;
}

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
  VertexAttributes(OpenMesh::Attributes::Status);
  FaceAttributes(OpenMesh::Attributes::Status);
  EdgeAttributes(OpenMesh::Attributes::Status);
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

template<typename MeshT>
inline void deleteFace(MeshT& mesh, const typename MeshT::FaceHandle& fh)
{
  if(!mesh.status(fh).deleted()) mesh.delete_face(fh,false);
}

typedef OpenMesh::TriMesh_ArrayKernelT<SurfaceTraits> Surface;

int main(int argc, char **argv)
{
  Surface sf;
  Surface::VertexHandle vh1[10];
  Surface::FaceHandle fh[13];
  std::cout << "Create Map" << std::endl;
  vh1[0] = sf.add_vertex(Surface::Point(0.,1.,0.));
  vh1[1] = sf.add_vertex(Surface::Point(4.,1.,0.));
  vh1[2] = sf.add_vertex(Surface::Point(2.,3.,0.));
  vh1[3] = sf.add_vertex(Surface::Point(6.,3.,0.));
  fh[0] = sf.add_face(vh1[0], vh1[1], vh1[2]);
  fh[1] = sf.add_face(vh1[1], vh1[3], vh1[2]);

  std::cout << "add 4" << std::endl;
  deleteFace(sf,fh[1]);
  vh1[4] = sf.add_vertex(Surface::Point(4.,2.,0.));
  fh[2] = sf.add_face(vh1[4],vh1[3],vh1[2]);

  std::cout << "add 5" << std::endl;
  deleteFace(sf,fh[0]);
  deleteFace(sf,fh[1]);
  vh1[5] = sf.add_vertex(Surface::Point(3.5,1.5,0.));
  fh[3] = sf.add_face(vh1[5],vh1[4],vh1[2]);

  std::cout << "add 6" << std::endl;
  deleteFace(sf,fh[1]);
  vh1[6] = sf.add_vertex(Surface::Point(4.5,1.5,0.));
  fh[4] = sf.add_face(vh1[6],vh1[3],vh1[4]);
  fh[5] = sf.add_face(vh1[6],vh1[4],vh1[5]);

  std::cout << "update 0" << std::endl;
  deleteFace(sf,fh[0]);
  fh[6] = sf.add_face(vh1[0],vh1[5],vh1[2]);

  std::cout << "add 7" << std::endl;
  deleteFace(sf,fh[0]);
  vh1[7] = sf.add_vertex(Surface::Point(3.,1.,0.));
  fh[7] = sf.add_face(vh1[7],vh1[5],vh1[0]);

  std::cout << "update 1" << std::endl;
  deleteFace(sf,fh[0]);
  deleteFace(sf,fh[1]);
  fh[8] = sf.add_face(vh1[1],vh1[6],vh1[5]);
  fh[9] = sf.add_face(vh1[1],vh1[5],vh1[7]);

  std::cout << "add 8" << std::endl;
  vh1[8] = sf.add_vertex(Surface::Point(2.,0.,0.));
  fh[10] = sf.add_face(vh1[8],vh1[1],vh1[7]);

  std::cout << "add 9" << std::endl;
  vh1[9] = sf.add_vertex(Surface::Point(6.,0.,0.));
  fh[11] = sf.add_face(vh1[9],vh1[6],vh1[1]);
  fh[12] = sf.add_face(vh1[9],vh1[1],vh1[8]);

  sf.garbage_collection();

  Surface::HalfedgeIter he_it;
  for(he_it=sf.halfedges_begin(); he_it!=sf.halfedges_end(); ++he_it)
  {
    //std::cout << "Edge: " << *he_it << std::endl;
    //std::cout << "v0: " << sf.to_vertex_handle(*he_it) << std::endl;
    //std::cout << "v1: " << sf.from_vertex_handle(*he_it) << std::endl;
  }

  Surface::EdgeIter e_it;
  for(e_it=sf.edges_begin(); e_it!=sf.edges_end(); ++e_it)
  {
    std::cout << "Edge: " << *e_it << std::endl;
    Surface::VertexHandle v1 = sf.to_vertex_handle(sf.halfedge_handle(*e_it,0));
    Surface::VertexHandle v2 = sf.to_vertex_handle(sf.halfedge_handle(*e_it,1));
    std::cout << "v1: " << sf.point(v1).transpose() << std::endl;
    std::cout << "v2: " << sf.point(v2).transpose() << std::endl;
    std::cout << "v1 " << (sf.point(v1) > sf.point(v2) ? ">" : "<") << " v2" << std::endl;
  }
  

  //OpenMesh::IO::write_mesh(sf, "/home/goa-sf/Desktop/test.ply");
  std::cout << "Hallo" << std::endl;
  return 0;
}

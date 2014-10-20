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

#include <iostream>
#include <chrono>

#include <boost/program_options.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/PolygonMesh.h>


#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

//#include "cob_3d_mapping_common/sensor_model.h"
#include "cob_surface/conversion.h"
#include "cob_surface/sensors.h"

std::string file_in_;
std::string file_out_;

int readOptions(int argc, char** argv)
{
  using namespace boost::program_options;
  options_description options("Options");
  options.add_options()
    ("help,h", "produce help message")
    ("in,i", value<std::string>(&file_in_), "input folder with data points")
    ("out,o", value<std::string>(&file_out_), "out file with data points")
    ;

  positional_options_description p_opt;
  p_opt.add("in", 1).add("out",1);
  variables_map vm;
  store(command_line_parser(argc, argv)
        .options(options).positional(p_opt).run(), vm);
  notify(vm);

  if(vm.count("help") || argc == 1)
  { std::cout << options << std::endl; return(-1); }

  return 0;
}


int main(int argc, char** argv)
{
  using namespace std::chrono;

  if(readOptions(argc, argv)<0) return 0;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    input(new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::PCDReader r;
  r.read(file_in_, *input);

  system_clock::time_point start, end;
  system_clock::duration elapsed;
  
  start = system_clock::now();
  pcl::PolygonMesh pcl_mesh;
  pcl::OrganizedFastMesh <pcl::PointXYZRGB> ofm;
  ofm.setInputCloud(input);
  ofm.setTriangulationType(pcl::OrganizedFastMesh<pcl::PointXYZRGB>
                           ::TRIANGLE_ADAPTIVE_CUT);
  ofm.reconstruct(pcl_mesh);
  end = system_clock::now();
  elapsed = duration_cast<microseconds>(end-start);
  std::cout << "OrganizedFastMesh took " << elapsed.count()/1000000. <<" sec"<<std::endl;

  typedef OpenMesh::TriMesh_ArrayKernelT<> MeshT;
  MeshT mesh;

  start = system_clock::now();
  cob_surface::Conversion<cob_surface::Kinect>
    ::pointCloud2Mesh<pcl::PointXYZRGB,MeshT>(input,mesh);

  end = system_clock::now();
  elapsed = duration_cast<microseconds>(end-start);
  std::cout << "My Meshing took " << elapsed.count()/1000000. <<" sec"<<std::endl;


  OpenMesh::IO::write_mesh(mesh,file_out_);
  return 0;
}

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

#include <OpenMesh/Core/IO/MeshIO.hh>

//#include "cob_3d_mapping_common/sensor_model.h"
//#include "cob_surface/conversion.h"
//#include "cob_surface/sensors.h"
#include "cob_surface/surface.h"
//#include "cob_surface/policies.h"
#include "cob_surface/merge.h"

int main(int argc, char** argv)
{
  using namespace std::chrono;

  if(0)//argc<4)
  {
    std::cout<<"Usage: test_mesh_sweepline input_1.ply input_2.ply output.ply"<<std::endl;
    return 0;
  }

  system_clock::time_point start, end;
  system_clock::duration elapsed;

  cob_surface::Surface sf1, sf2;
  //OpenMesh::IO::read_mesh(sf1, argv[1]);
  //OpenMesh::IO::read_mesh(sf2, argv[2]);
  std::string in1("/home/goa-sf/benchmark/meshes/random/random_mesh_flat_01.ply");
  std::string in2("/home/goa-sf/benchmark/meshes/random/random_mesh_flat_02.ply");
  std::string out("/home/goa-sf/benchmark/meshes/random/out.ply");
  OpenMesh::IO::read_mesh(sf1, in1);
  OpenMesh::IO::read_mesh(sf2, in2);
  

  start = system_clock::now();

  cob_surface::Merge<cob_surface::Surface> merge;
  cob_surface::transformToProjSpace(&sf1);
  cob_surface::transformToProjSpace(&sf2);
  merge.initialize(&sf1,&sf2);
  merge.compute(&sf2);

  end = system_clock::now();
  elapsed = duration_cast<microseconds>(end-start);
  std::cout << "Merging took " << elapsed.count()/1000000. <<" sec"<<std::endl;

  OpenMesh::IO::write_mesh(sf2,out);
  return 0;
}

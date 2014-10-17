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

#include <iostream>
//#include <utility>
//#include <set>
//#include <unordered_map>
//#include <queue>
#include <random>
#include <functional>
#include <chrono>
#include <ctime>

#include "cob_surface/sweepline.h"


struct Vertex
{
  double x;
  double y;
};

std::ostream &operator<<(std::ostream &os, Vertex const &v) {
  return os << "(" << v.x <<","<< v.y << ")";
}

std::ostream &operator<<(std::ostream &os, std::pair<double,double> const &p) {
  return os << "(" << p.first <<","<< p.second << ")";
}

/******************************************************************************
 * Data Dependent Definitions:
 ******************************************************************************/

typedef std::pair<Vertex,Vertex> Segment;

struct LineTraits
{
  typedef std::pair<Vertex,Vertex> DataT;
  typedef std::pair<double,double> StateT;
};


struct LinePolicy
{
  // returns true if segment a comes before b at sweepline state
  static bool compare(const Segment& a, const Segment& b,
                      const std::pair<double,double>& state)
  {
    double y = state.first;
    //std::cout << "State:" << y << std::endl;
    if (a.first.y == y and b.first.y == y) // a & b have same start point
    {
      if (a.first.x == b.first.x)
        return a.second.x < b.second.x;
      else
        return a.first.x < b.first.x;
    }

    // intersection of y at a:
    double am = (a.first.y - y)/(a.first.y - a.second.y);
    double ax = am * (a.second.x - a.first.x) + a.first.x;
    // intersection of y at b:
    double bm = (b.first.y - y)/(b.first.y - b.second.y);
    double bx = bm * (b.second.x - b.first.x) + b.first.x;
    if (ax == bx)
    {
      std::cout << "swap point " << std::endl;
      return a.second.x < b.second.x;
    }
    return ax < bx;
  }

  // create a fake data object for state localization (vertical line)
  static Segment dataForLocalization(const std::pair<double,double>& state)
  {
    return Segment({state.second,state.first},{state.second,state.first-1.f});
  }

  //http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
  static bool swapCheck(const Segment& a, const Segment& b,
                        std::pair<double,double>& state)
  {
    double sa_x = a.first.x - a.second.x;
    double sa_y = a.first.y - a.second.y;
    double sb_x = b.first.x - b.second.x;
    double sb_y = b.first.y - b.second.y;

    double denom = sa_x * sb_y - sb_x * sa_y;
    if (denom == 0)
      return false; // Collinear
    bool denomPositive = denom > 0;

    double sab_x = a.second.x - b.second.x;
    double sab_y = a.second.y - b.second.y;
    double s_numer = sa_x * sab_y - sa_y * sab_x;
    if ((s_numer < 0) == denomPositive)
      return false; // No collision

    double t_numer = sb_x * sab_y - sb_y * sab_x;
    if ((t_numer < 0) == denomPositive)
      return false; // No collision

    if (((s_numer > denom) == denomPositive) ||
        ((t_numer > denom) == denomPositive))
      return false; // No collision

    double t = t_numer / denom;
    state = std::make_pair(a.second.y + (t*sa_y), a.second.x + (t*sa_x));
    return true;
  }
};

int main(int argc, char **argv)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1.,1.);
  auto rng = std::bind(dist,gen); // use rng() to generate random number

  std::vector<Segment> segments;
  
  auto start =  std::chrono::system_clock::now();
  for (int i=0; i<1000; ++i)
  {
    double ay = rng();
    double by = rng();
    if (ay > by)
      segments.push_back( Segment({rng(),ay},{rng(),by}) );
    else
      segments.push_back( Segment({rng(),by},{rng(),ay}) );
  }
  auto end =  std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
  std::cout << elapsed.count()/1000000. << std::endl;
  /*
  std::vector<Segment> segments{
    Segment({3.5,6.0}, {6.5,0}),
      Segment({5.0,4.5}, {2.0,1.5}),
      Segment({10.,5.0}, {0,1.})
      };
  */

  start =  std::chrono::system_clock::now();

  std::vector<SweepLine::DataId> d_ids;
  SweepLine::SweepLineProcess<Segment,std::pair<double,double>, LinePolicy> sl;
  sl.addAllData(segments.begin(), segments.end(), segments.size(), d_ids);
  for (int i=0; i<d_ids.size(); ++i)
  {
    Segment* s = &sl.getData(d_ids[i]);
    // start event
    sl.addEvent(std::make_pair(s->first.y, s->first.x), {d_ids[i]}, {});
    // end event
    sl.addEvent(std::make_pair(s->second.y, s->second.x), {}, {d_ids[i]});
  }

  Segment* left;
  Segment* right;
  int i = 0;
  while(sl.nextEvent() )//&& i<30)
  {
    //std::cout << i++ << std::endl;
/*
    std::cout << "id2data:" << std::endl;
    for(auto it = sl.id2data.begin(); it != sl.id2data.end(); ++it)
    {
      std::cout << it->first << ": " << it->second << std::endl;
    }
    std::cout << "data2node:" << std::endl;
    for(auto it = sl.data2node.begin(); it != sl.data2node.end(); ++it)
    {
      std::cout << it->first << ": " << it->second->id << std::endl;
    }
*/
  }

  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
  std::cout << elapsed.count()/1000000. << std::endl;
  return 0;
}

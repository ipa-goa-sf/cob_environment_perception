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
 * \date Date of creation: 11/2014
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

#ifndef COB_SURFACE_ADVANCING_FRONT_H
#define COB_SURFACE_ADVANCING_FRONT_H

/*
 * Domiter, Vid, and Borut Žalik. "Sweep‐line algorithm for constrained Delaunay
 * triangulation." International Journal of Geographical Information Science
 * 22.4 (2008): 449-462.
 *
 */

#include <set>

#include "cob_surface/sweepline.h"

namespace SweepLine
{
  /**
   * @class AdvancingFront
   * @brief maintains the current edges of the triangulated surface
   *
   * A new constrained delaunay triangulation is constructed while iteratively
   * inserting new points and edges. The insertation process has to be performed
   * in an ordered way.
   *
   * @tparam Traits - defines surface type and its elements (VertexT,..)
   * @tparam Policy - provides policies to manipulate the specified surface
   *
   */
  template<typename Traits, typename Policy>
  class AdvancingFront
  {
  public:
    typedef typename Traits::SurfaceT SurfaceT;
    typedef typename Traits::VertexHandle VertexHandle;
    typedef typename Traits::SurfaceT::Point PointT;

    typedef typename std::set<VertexHandle,DelegateCompare>::iterator NodeIter;
    

    // delegates the compare operation of BST to definition by policy
    struct DelegateCompare
    {
      DelegateCompare(SurfaceT* sf) : sf_(sf) { }
      bool operator() (const VertexHandle& a, const VertexHandle& b) {
        return Policy::vertexCompare(a,b,sf_);
      }

      SurfaceT* sf_;
    };

  public:
    AdvancingFront(SurfaceT* sf)
      : bst_(DelegateCompare(sf))
      , sf_(sf)
    { }

    ~AdvancingFront() { }

    /** 
     * Provide a list of vertices that form the initial advancing front
     * This list should contain at least 2 vertices that encapsulate
     * all vertices (orthogonal to the insert direction)
     * that are going to be inserted in the process. Hence:
     * v_1 = x_min - alpha(x_max - x_min)
     * v_2 = x_max + alpha(x_max - x_min)
     * with alpha > 0 (eg. 0.3)
     * 
     * @param begin iterator to first vertex
     * @param end  iterator past the last element (not being used)
     */
    template<typename IteratorT>
    inline void initialize(const IteratorT& begin, const IteratorT& end) {
      bst_.insert(begin,end);
    }

    inline void initialize(const VertexHandle& vh1, const VertexHandle& vh2) {
      bst_.insert(vh1);
      bst_.insert(vh2);
    }

    /** 
     * Provide a list of vertices who's face should be deleted.
     * This step can be used to undo the insertation of auxiliary vertices
     * introduced with the initialization step.
     * 
     * @param begin iterator to first vertex
     * @param end  iterator past the last element (not being used)
     */
    template<typename IteratorT>
    void finalize(const IteratorT& begin, const IteratorT& end);

    void insertVertex(const VertexHandle& v);

    void insertEdge(const VertexHandle& v1, const VertexHandle& v2);

    inline float vertexPos(const PointT& v, const PointT& e, const PointT& p) {
      return e.cross(v-p).dot(p);
    }

    inline HalfedgeHandle advance(const HalfedgeHandle& heh) {
      return sf_->next_halfedge_handle(sf_->opposite_halfedge_handle(heh));
    }

  private:

    std::set<VertexHandle,DelegateCompare> bst_;
    SurfaceT* sf_;

  };
}

#include "cob_surface/impl/advancing_front.hpp"

#endif

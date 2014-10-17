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

#ifndef COB_SURFACE_SWEEPLINE_HPP
#define COB_SURFACE_SWEEPLINE_HPP

template<typename T, typename StateT, typename PolicyT>
template<typename InputIterator>
void SweepLine::SweepLineProcess<T,StateT,PolicyT>::addAllData(
  const InputIterator& first,
  const InputIterator& last,
  size_t n,
  std::vector<DataId>& out_data_ids)
{
  DataId id = 0;

  bst_.clear();
  event_schedule_ = std::priority_queue<Event>();
  data_.clear();
  data_.resize(n+1); // last place is for localization data
  data2node_.clear();
  data2node_.resize(n);
  node2data_.clear();
  node2data_.resize(n+1); // keep a place for localization data
  locate_node_id_ = NodeId(n);
  node2data_[locate_node_id_] = DataId(n);
  adjacent_data_.clear();
  out_data_ids.clear();
  out_data_ids.resize(n);

  // there might be a better way for doing this:
  for(InputIterator it = first; it != last; ++it)
  {
    data_[id] = *it;
    out_data_ids[id] = id++;
  }
}

template<typename T, typename StateT, typename PolicyT>
bool SweepLine::SweepLineProcess<T,StateT,PolicyT>::nextEvent()
{
  if (event_schedule_.empty()) return false;

  // order: lnd <-> lnd_new <-> more nodes <-> rnd_new <-> rnd
  NodeIter lnd, lnd_new, rnd_new, rnd;
  DataId d1, d2, d3, d4;
  DataPairId pId;

  std::pair<bool,bool> valid_neighbors;
  StateT state_new;
  Event e = event_schedule_.top();
  state_ = e.state;
  static int n_intersections = 0;
  // process swap event (eg. line intersection)
  if (e.swap_event)
  {
    //std::cout << "Process SwapEvent " << e.state << std::endl;
    // order before swap: lnd <-> lnd_new <-> rnd_new <-> rnd
    // order after swap: lnd <-> rnd_new <-> lnd_new <-> rnd
    d2 = *(e.to_insert.begin());
    d3 = *(e.to_remove.begin());
    lnd_new = getNode(d2); // do we really need this ???
    rnd_new = getNode(d3); // separate event for swap stores node directly
    rnd = rightNeighbor(rnd_new);

    if (lnd_new != bst_.begin())
    {
      lnd = leftNeighbor(lnd_new);
      processAdjacency(lnd, rnd_new);
    }

    if (rnd != bst_.end()) processAdjacency(lnd_new, rnd);

    swap(lnd_new, rnd_new, d2, d3);

    // done with this event
    event_schedule_.pop();
    ++n_intersections;

    return true;
  }

  // process remove list or locate for insertation position
  if (!e.to_remove.empty())
  {
    //std::cout << "Process RemoveEvent "<< e.state << std::endl;
    valid_neighbors = remove(e.to_remove.begin(), e.to_remove.end(), lnd, rnd);
  }
  else
  {
    valid_neighbors = locate(e.state, lnd, rnd);
  }

  if (!e.to_insert.empty())
  {
    // process insert list
    //std::cout << "Process InsertEvent " << e.state << std::endl;
    insert(rnd, e.to_insert.begin(), e.to_insert.end(), lnd_new, rnd_new);

    if (valid_neighbors.first ) processAdjacency(lnd, lnd_new);
    if (valid_neighbors.second) processAdjacency(rnd_new, rnd);
  }
  else if (valid_neighbors.first && valid_neighbors.second)
  {
    // something was removed and nothing inserted
    processAdjacency(lnd, rnd);
  }

  event_schedule_.pop();
  if (event_schedule_.empty())
    std::cout << "Intersections: " << n_intersections << std::endl;
  return true;
}

template<typename T, typename StateT, typename PolicyT>
std::pair<bool,bool> SweepLine::SweepLineProcess<T,StateT,PolicyT>::locate(
  const StateT& state, NodeIter& left, NodeIter& right)
{
  if (bst_.empty())
  {
    left  = bst_.begin();
    right = bst_.end();
    return std::make_pair(false,false);
  }

  data_.back() = PolicyT::dataForLocalization(state);
  right = bst_.upper_bound(locate_node_id_);
  bool has_left_neighbor = (right != bst_.begin());
  if(has_left_neighbor) left = leftNeighbor(right);
  return std::make_pair(has_left_neighbor, right!=bst_.end());
}

template<typename T, typename StateT, typename PolicyT>
void SweepLine::SweepLineProcess<T,StateT,PolicyT>::insert(
  const NodeIter& position,
  const std::vector<DataId>::iterator& first,
  const std::vector<DataId>::iterator& last,
  NodeIter& left, NodeIter& right)
{
  NodeIter nd_new = position;
  DataId d_id;
  NodeId n_id;
  for(std::vector<DataId>::iterator it = first; it != last; ++it)
  {
    d_id = *it;
    n_id = *it; //generateNodeId();
    node2data_[n_id] = d_id;
    nd_new = bst_.insert(nd_new, n_id);
    data2node_[d_id] = nd_new++;
  }

  left = getNode(*first);
  right = getNode(d_id);
}

template<typename T, typename StateT, typename PolicyT>
std::pair<bool,bool> SweepLine::SweepLineProcess<T,StateT,PolicyT>::remove(
  const std::vector<DataId>::iterator& first,
  const std::vector<DataId>::iterator& last,
  NodeIter& left, NodeIter& right)
{
  NodeIter nd = bst_.end();
  for(typename std::vector<DataId>::iterator it = first; it != last; ++it)
  {
    DataId d_id = *it;
    nd = getNode(d_id);
    bst_.erase(nd++);
  }
  right = nd;
  bool valid_left;
  if(valid_left = (right != bst_.begin())) left = --nd;
  else left = nd;

  return std::make_pair(valid_left, right != bst_.end());
}

template<typename T, typename StateT, typename PolicyT>
void SweepLine::SweepLineProcess<T,StateT,PolicyT>::swap(
  const NodeIter& a, const NodeIter& b, DataId data_a, DataId data_b)
{
  data2node_[data_a] = b;
  data2node_[data_b] = a;
  node2data_[*a] = data_b;
  node2data_[*b] = data_a;
}


template<typename T, typename StateT, typename PolicyT>
void SweepLine::SweepLineProcess<T,StateT,PolicyT>::processAdjacency(
  const NodeIter& a, const NodeIter& b)
{
  DataId da = getDataId(*a);
  DataId db = getDataId(*b);
  DataPairId pid1 = makeDataPairId(da,db);
  DataPairId pid2 = makeDataPairId(db,da);
  StateT s;
  //std::cout << "SwapCheck: "<< da << " - " << db << std::endl;
  if (adjacent_data_.count(pid1) == 0 && adjacent_data_.count(pid2) == 0)
  {
    if(PolicyT::swapCheck(getData(da),getData(db),s))
    {
      adjacent_data_.insert(pid1);
      addSwapEvent(s, da, db);
    }
  }
}


#endif

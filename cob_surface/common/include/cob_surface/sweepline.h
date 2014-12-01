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

#ifndef COB_SURFACE_SWEEPLINE_H
#define COB_SURFACE_SWEEPLINE_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <set>
#include <algorithm>


namespace SweepLine
{
  typedef uint32_t NodeId;
  typedef uint32_t DataId;
  typedef uint64_t DataPairId;

  template<typename Traits>
  struct Event
  {
    typename Traits::StateT state;
    std::vector<DataId> to_insert;
    std::vector<DataId> to_remove;
    bool swap_event;
  };

  inline DataPairId makeDataPairId(DataId a, DataId b) {
    return (DataPairId(a) << 32) | DataPairId(b);
  }

  template<typename T>
  T leftNeighbor(T pos) { return --pos; }

  template<typename T>
  T rightNeighbor(T pos) { return ++pos; }

  /**
   * @class SweepLineProcess
   * @brief performs a sweepline process over assigned data
   *
   * @tparam Traits - defines DataT and StateT
   * @tparam Policy - policies for compare and fake data
   *
   */
  template<typename Traits, typename Policy>
  class SweepLineProcess
  {
  public:
    typedef typename Traits::DataT DataT;
    typedef typename Traits::StateT StateT;
    typedef Event<Traits> Event;

    friend inline const bool operator< (const Event& lhs, const Event& rhs)
    {
      return Policy::stateCompare(lhs.state, rhs.state)
    }

    // delegates the compare operation of BST to definition by policy
    struct DelegateCompare
    {
      DelegateCompare(SweepLineProcess<Traits,Policy>* sl) : p(sl) { }
      bool operator() (NodeId a, NodeId b)
      {
        return Policy::dataCompare(p->getData(p->getDataId(a)),
                                   p->getData(p->getDataId(b)), p->state_);
      }
      /*
      bool operator() (DataId a, DataId b)
      {
        return Policy::dataCompare(p->getData(a), p->getData(b));
        }*/

      SweepLineProcess<Traits,Policy>* p; ///< pointer to parent
    };

    typedef typename std::set<NodeId,DelegateCompare>::iterator NodeIter;

  public:
    SweepLineProcess() : bst_(DelegateCompare(this)) { }

    /** 
     * Clear assigned data and event_schedule. Allocate space for new data.
     * 
     * @param n_data - number of new data elements
     */
    void reset(unsigned int n_data);

    /** 
     * Add a new data element.
     * 
     * @param data - the new element to add
     * 
     * @return DataId of the added element
     */
    inline DataId addData(const DataT& data)
    {
      data_[next_id_] = data;
      out_data_ids_[next_id_] = next_id_++;
    }

    /** 
     * Copy all data to the SweepLineProcess and returns a list of
     * data ids. Data can be accessed via getData()
     * 
     * @param first - iterator to the first data element
     * @param last - iterator past the last data element
     * @param n - number of all elements
     * @param out_data_ids - OUT: list of DataId
     */
    template<typename IteratorT>
    void addAllData(const IteratorT& begin,
                    const IteratorT& end,
                    size_t n,
                    std::vector<DataId>& out_data_ids);

    /** 
     * Copy event to event schedule
     * 
     * @param event - the event to add
     * @param data_is_sorted - set to false if sweepline needs to sort the data
     *    of to_insert and to_remove before the event is added.
     */
    inline void addEvent(const Event& event, bool data_is_sorted = false)
    {
      if(!data_is_sorted)
      {
        Event e = event;
        std::sort(e.to_insert.begin(), e.to_insert.end(), DelegateCompare(this));
        std::sort(e.to_remove.begin(), e.to_remove.end(), DelegateCompare(this));
        event_schedule_.push(e);
      }
      else
      {
        event_schedule_.push(event);
      }
    }

    /**
     * Add event for data start point and/or data end point
     *
     * @param event_state - state when event needs to be processed
     * @param data_to_insert - list of DataIds to insert at the event
     * @param data_to_remove - list of DataIds to remove at the event
     */
    inline void addEvent(const StateT& event_state,
                         const std::vector<DataId>& data_to_insert,
                         const std::vector<DataId>& data_to_remove)
    {
      event_schedule_.push(Event({event_state,data_to_insert,data_to_remove,false}));
    }

    /**
     * Create event to swap data_a with data_b on sweep line
     *
     * @param event_state - state when event needs to be processed
     * @param data_a - left DataId
     * @param data_b - right DataId
     */
    inline void addSwapEvent(const StateT& event_state,
                             DataId data_a, DataId data_b)
    {
      event_schedule_.push(Event({event_state, {data_a}, {data_b}, true}));
    }

    /**
     * Process next event in event schedule
     *
     * @return False if all events have been processed
     */
    bool nextEvent();

    /** 
     * Returns data by DataId
     * 
     * @param id - query DataId
     * @return associated data T
     */
    inline DataT& getData(const DataId& id) { return data_[id]; }

    /** 
     * Get the event that was currently processed
     * 
     * @return the event
     */
    inline Event& getCurrentEvent() { return current_event_; }

    /** 
     * Get the DataId left to the currently processed event
     * 
     * @param id - out
     * 
     * @return - False if there is nothing to the left (DataId invalid)
     */
    inline bool getLeftDataId(DataId& id)
    {
      if (!left_neighbor_.first) return false;
      id = getDataId(left_neighbor_.second);
      return true;
    }

    /** 
     * Get the DataId right to the currently processed event
     * 
     * @param id - out
     * 
     * @return - False if there is nothing to the right (DataId invalid)
     */
    inline bool getRightDataId(DataId& id)
    {
      if (!right_neighbor_.first) return false;
      id = getDataId(right_neighbor_.second);
      return true;
    }

  private:
    /** 
     * Locate left and right of state in current sweepline
     * 
     * @param state - state for query
     * @param left - OUT: iterator of left node
     * @param right - OUT: iterator of right node
     * 
     * @return first/second is true if left/right is a valid iterator
     */
    std::pair<bool,bool> locate(const StateT& state,
                                NodeIter& left,
                                NodeIter& right);

    /** 
     * insert data in current sweepline at position
     * Note: all data to be inserted has common state
     * and is sorted from first < last
     * 
     * @param position - data is inserted before position
     * @param first - first data to be inserted
     * @param last - iterator when to stop insertation
     * @param left - OUT: first node on the left that was inserted
     * @param right  - OUT: last node on the right that was inserted
     */
    void insert(const NodeIter& position,
                const std::vector<DataId>::iterator& first,
                const std::vector<DataId>::iterator& last,
                NodeIter& left,
                NodeIter& right);

    /** 
     * remove data from current sweepline
     * Note: all data to be removed has common state
     * 
     * @param first - first data to remove
     * @param last - iterator when to stop removal
     * @param left - OUT: left node that became adjacent
     * @param right - OUT: right node that became adjacent
     *
     * @return first/second is true if left/right is a valid iterator
     */
    std::pair<bool,bool> remove(const std::vector<DataId>::iterator& first,
                                const std::vector<DataId>::iterator& last,
                                NodeIter& left,
                                NodeIter& right);

    /** 
     * performs a swap on association containers
     * 
     * @param a - node iterator a
     * @param b - node iterator b
     * @param data_a - Data ID a
     * @param data_b - Data ID b
     */
    void swap(const NodeIter& a, const NodeIter& b, DataId data_a, DataId data_b);

    void processAdjacency(const NodeIter& a, const NodeIter& b);

    NodeId generateNodeId()
    {
      static NodeId s_nId = 1;
      return s_nId++;
    }

    /// gets node iterator from data ID
    inline NodeIter& getNode(const DataId& d) { return data2node_[d]; }

    /// gets data ID from node ID
    inline DataId& getDataId(const NodeId& n) { return node2data_[n]; }

  private:
    /* Note: association containers are necessary, because the nodes in 
       std::set structure may not be modified (e.g. swap operation) */

    /// represents the currently active data on SL as binary search tree
    std::set<NodeId, DelegateCompare> bst_;
    /// queue of events left to process
    std::priority_queue<Event> event_schedule_;
    ///
    std::vector<DataT> data_;
    /// maintains data to bst node association
    std::vector<NodeIter> data2node_;
    /// maintaints bst node to data association
    std::vector<DataId> node2data_;
    /// keeps track of data adjacency to avoid creation of multiple events
    std::unordered_set<DataPairId> adjacent_data_;

    NodeId locate_node_id_;
    DataId next_id_;
    /// current state
    StateT state_;
    Event current_event_;
    std::pair<bool,NodeIter> left_neighbor_;
    std::pair<bool,NodeIter> right_neighbor_;
  };
}

#include "cob_surface/impl/sweepline.hpp"

#endif

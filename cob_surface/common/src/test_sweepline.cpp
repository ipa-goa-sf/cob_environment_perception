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
#include <utility>
#include <set>
#include <unordered_map>
#include <queue>
#include <random>
#include <functional>

struct Vertex
{
  float x;
  float y;
};

std::ostream &operator<<(std::ostream &os, Vertex const &v) {
  return os << "(" << v.x <<","<< v.y << ")";
}

std::ostream &operator<<(std::ostream &os, std::pair<float,float> const &p) {
  return os << "(" << p.first <<","<< p.second << ")";
}


template<typename T, typename PolicyT, typename StateT = float>
class SweepLine
{
public:
  // Node of Binary Search Tree (BST, set<BstNode>), data T wrapper
  struct BstNode {
    T* p_data;
  };

  struct Event {
    StateT state;
    std::vector<T*> to_insert;
    std::vector<T*> to_remove;
    bool swap_event;
  };

  friend inline const bool operator< (const Event& lhs, const Event& rhs) {
    return lhs.state < rhs.state; }

  friend inline const bool operator> (const Event& lhs, const Event& rhs) {
    return operator< (rhs, lhs); }

  friend inline const bool operator<= (const Event& lhs, const Event& rhs) {
    return !operator> (lhs, rhs); }

  friend inline const bool operator>= (const Event& lhs, const Event& rhs) {
    return !operator< (lhs, rhs); }

  // delegates the compare operation of BST to PolicyT class
  struct DelegateCompare {
    DelegateCompare(SweepLine<T,PolicyT,StateT>* sl) : parent(sl) { }

    bool operator() (const BstNode& a, const BstNode& b) {
      return PolicyT::compare(*(a.p_data), *(b.p_data), parent->state); }

    SweepLine<T,PolicyT,StateT>* parent;
  };

  typedef typename std::set<BstNode,DelegateCompare>::iterator NodeIter;

public:
  SweepLine() : bst(DelegateCompare(this)) { }

  inline void addEvent(const StateT& event_state,
                       const std::vector<T*>& data_to_insert,
                       const std::vector<T*>& data_to_remove)
  {
    event_schedule.push(
      Event({event_state, data_to_insert, data_to_remove, false})
      );
  }

  inline void addSwapEvent(const StateT& event_state,
                           T* data_a, T* data_b)
  {
    event_schedule.push(
      Event({event_state, {data_a}, {data_b}, true})
      );
    std::cout << "New Swap Event: " << event_state << std::endl;
  }

  bool nextEvent()
  {
    if (event_schedule.empty()) return false;

    // order: lnd <-> lnd_new <-> more nodes <-> rnd_new <-> rnd
    NodeIter lnd; // left node
    NodeIter rnd; // right node
    NodeIter lnd_new; // left new node
    NodeIter rnd_new; // right new node
    std::pair<bool,bool> valid_neighbors;
    StateT state_new;
    Event e = event_schedule.top();
    setState(e.state);

    // process swap event (eg. line intersection)
    if (e.swap_event)
    {
      lnd = data_map.at( *(e.to_insert.begin()) );
      rnd = data_map.at( *(e.to_remove.begin()) );
      std::cout << "SwapEvent 1:" << lnd->p_data <<" "<< rnd->p_data<< std::endl;
      swap(lnd, rnd);
      std::cout << "SwapEvent 2:" << lnd->p_data <<" "<< rnd->p_data<< std::endl;

      if(lnd != bst.begin()) // if not the first element
      {
        // save swapped node as lnd_new
        // with lnd advance one more to the left (order: lnd <-> lnd_new)
        lnd_new = lnd--;
        std::cout << "SwapCheck L:" << lnd->p_data <<" "<<lnd_new->p_data<< std::endl;
        if(PolicyT::swapCheck(*(lnd->p_data), *(lnd_new->p_data), state_new))
          addSwapEvent(state_new, lnd->p_data, lnd_new->p_data);
      }

      rnd_new = rnd++;
      if(rnd != bst.end())
      {
        std::cout << "SwapCheck R:" << rnd_new->p_data <<" "<<rnd->p_data<< std::endl;
        if(PolicyT::swapCheck(*(rnd_new->p_data), *(rnd->p_data), state_new))
          addSwapEvent(state_new, rnd_new->p_data, rnd->p_data);;
      }

      // done with this event
      event_schedule.pop();
      return true;
    }

    // process remove list or locate for insertation position
    if (!e.to_remove.empty())
    {
      valid_neighbors = remove(e.to_remove.begin(), e.to_remove.end(), lnd, rnd);
    }
    else
    {
      valid_neighbors = locate(e.state, lnd, rnd);
    }

    if(!e.to_insert.empty())
    {
      // process insert list
      insert(rnd, e.to_insert.begin(), e.to_insert.end(), lnd_new, rnd_new);
      if (valid_neighbors.first) // has left neighbor
      {
        std::cout << "Insert Swap Check 1" << std::endl;
        // check for intersection lnd & lnd_new
        if(PolicyT::swapCheck(*(lnd->p_data), *(lnd_new->p_data), state_new))
          addSwapEvent(state_new, lnd->p_data, lnd_new->p_data);
      }
      if (valid_neighbors.second) // has right neighbor
      {
        std::cout << "Insert Swap Check 2" << std::endl;
        // check for intersection rnd & rnd_new
        if(PolicyT::swapCheck(*(rnd_new->p_data), *(rnd->p_data), state_new))
          addSwapEvent(state_new, rnd_new->p_data, rnd->p_data);
      }
    }
    else if(valid_neighbors.first && valid_neighbors.second)
    {
       // something was removed and nothing inserted
      std::cout << "Remove Swap Check" << std::endl;
      // check for intersection lnd & rnd
      if(PolicyT::swapCheck(*(lnd->p_data), *(rnd->p_data), state_new))
        addSwapEvent(state_new, lnd->p_data, rnd->p_data);
    }
    // done with this event
    event_schedule.pop();
    return true;
  }

  std::pair<bool,bool> locate(const StateT& state, NodeIter& left, NodeIter& right)
  {
    if (bst.empty())
    {
      std::cout << "BST empty" << std::endl;
      left  = bst.begin();
      right = bst.end();
      return std::make_pair(false,false);
    }

    T data = PolicyT::fakeData(state);
    BstNode fake_nd = BstNode({&data});
    left = bst.upper_bound(fake_nd);
    bool has_left_neighbor = (left != bst.begin());
    if(has_left_neighbor)
      right = left--;
    else
      right = left;
    return std::make_pair(has_left_neighbor,right!=bst.end());
  }

  void insert(const NodeIter& position,
              const typename std::vector<T*>::iterator& first,
              const typename std::vector<T*>::iterator& last,
              NodeIter& first_insert,
              NodeIter& last_insert)
  {
    /* Assumptions:
       1: all T* share the same starting point
          -> all T* will be inserted adjacent to each other
          -> left and right neighbors can be found by checking only the first T*
       2: first to last in ascending order
          -> T*_2 can be inserted right after T*_1
     */
    // Since C++11: set.insert(hint,value)
    // hint has to point behind the element that will be inserted, not before
    NodeIter nd_iter_new = position;
    for(typename std::vector<T*>::iterator it = first; it != last; ++it)
    {
      BstNode nd = BstNode({*it});
      nd_iter_new = bst.insert(nd_iter_new, nd);
      data_map[*it] = nd_iter_new;
      ++nd_iter_new;
    }

    first_insert = data_map.at(*first);
    last_insert = --nd_iter_new;
  }

  std::pair<bool,bool> remove(const typename std::vector<T*>::iterator& first,
                              const typename std::vector<T*>::iterator& last,
                              NodeIter& left, NodeIter& right)
  {
    NodeIter nd = bst.end();
    for(typename std::vector<T*>::iterator it = first; it != last; ++it)
    {
      nd = data_map.at(*it);
      std::cout << "Remove: " << *it << " " << nd->p_data << std::endl;
      data_map.erase(*it);
      bst.erase(nd++);
    }
    right = nd;
    left = --nd;
    return std::make_pair(left != bst.begin(), right != bst.end());
  }

  void swap(NodeIter& a, NodeIter& b)
  {
    /*std::cout << "a: "<< (a==bst.begin() ? "Begin ":"")
              <<  (a==bst.end() ? "End":"") <<std::endl;
    std::cout << "b: "<< (b==bst.begin() ? "Begin ":"")
    <<  (b==bst.end() ? "End":"") <<std::endl;*/

    T* bnew = a->p_data;
    T* anew = b->p_data;
    NodeIter tmp = a;
    NodeIter pos_a = a;
    NodeIter pos_b = b;
    //std::cout << "a: " << &(*pos_a) << std::endl;
    //std::cout << "b: " << &(*pos_b) << std::endl;
    bst.erase(tmp++); // delete a and advance to b
    bst.erase(tmp++); // delete b and advance past b
    pos_a = bst.insert(tmp,BstNode({anew})); // insert new a at old b
    //std::cout << "a: " << &(*pos_a) <<" "<< &(*a) << std::endl;
    pos_b = bst.insert(pos_a,BstNode({bnew})); // insert new b at old a
    //std::cout << "b: " << &(*pos_b) <<" "<< &(*b) << std::endl;

    //std::cout << "a: " << pos_a->p_data << " " << b->p_data << std::endl;
    //std::cout << "b: " << pos_b->p_data << " " << a->p_data << std::endl;
    data_map[pos_a->p_data] = pos_a;
    data_map[pos_b->p_data] = pos_b;
    a = pos_a;
    b = pos_b;
  }

  void setState(const StateT& state) { this->state = state; }

private:
  std::set<BstNode, DelegateCompare> bst;
  std::priority_queue<Event> event_schedule;
  std::unordered_map<T*,NodeIter> data_map;
  StateT state;
};


/******************************************************************************
 * Data Dependent Definitions:
 ******************************************************************************/

typedef std::pair<Vertex,Vertex> Segment;

struct LineTraits
{
  typedef std::pair<Vertex,Vertex> DataT;
  typedef std::pair<float,float> StateT;
};


struct LinePolicy
{
  // returns true if segment a comes before b at sweepline state
  static bool compare(const Segment& a, const Segment& b,
                      const std::pair<float,float>& state)
  {
    float y = state.first;
    std::cout << "State:" << y << std::endl;
    if (a.first.y == y and b.first.y == y) // a & b have same start point
    {
      if (a.first.x == b.first.x)
        return a.second.x < b.second.x;
      else
        return a.first.x < b.first.x;
    }

    // intersection of y at a:
    float am = (a.first.y - y)/(a.first.y - a.second.y);
    float ax = am * (a.second.x - a.first.x) + a.first.x;
    // intersection of y at b:
    float bm = (b.first.y - y)/(b.first.y - b.second.y);
    float bx = bm * (b.second.x - b.first.x) + b.first.x;
    if (ax == bx)
    {
      std::cout << "swap point " << std::endl;
      return a.second.x < b.second.x;
    }
    return ax < bx;
  }

  // create a fake data object for state localization (vertical line)
  static Segment fakeData(const std::pair<float,float>& state)
  {
    return Segment({state.second,state.first},{state.second,state.first-1.f});
  }

  //http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
  static bool swapCheck(const Segment& a, const Segment& b,
                        std::pair<float,float>& state)
  {
    float sa_x = a.first.x - a.second.x;
    float sa_y = a.first.y - a.second.y;
    float sb_x = b.first.x - b.second.x;
    float sb_y = b.first.y - b.second.y;

    float denom = sa_x * sb_y - sb_x * sa_y;
    if (denom == 0)
      return false; // Collinear
    bool denomPositive = denom > 0;

    float sab_x = a.second.x - b.second.x;
    float sab_y = a.second.y - b.second.y;
    float s_numer = sa_x * sab_y - sa_y * sab_x;
    if ((s_numer < 0) == denomPositive)
      return false; // No collision

    float t_numer = sb_x * sab_y - sb_y * sab_x;
    if ((t_numer < 0) == denomPositive)
      return false; // No collision

    if (((s_numer > denom) == denomPositive) ||
        ((t_numer > denom) == denomPositive))
      return false; // No collision

    float t = t_numer / denom;
    state = std::make_pair(a.second.y + (t*sa_y), a.second.x + (t*sa_x));
    return true;
  }
};

int main(int argc, char **argv)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.,1.);
  auto rng = std::bind(dist,gen); // use rng() to generate random number

  std::vector<Segment> segments;

  for (int i=0; i<100; ++i)
  {
    float ay = rng();
    float by = rng();
    if (ay > by)
      segments.push_back( Segment({rng(),ay},{rng(),by}) );
    else
      segments.push_back( Segment({rng(),by},{rng(),ay}) );
  }

  SweepLine<Segment,LinePolicy,std::pair<float,float> > sl;
  for (int i=0; i<segments.size(); ++i)
  {
    sl.addEvent(std::make_pair(segments[i].first.y, segments[i].first.x),
                { &segments[i] }, { }); // start event
    sl.addEvent(std::make_pair(segments[i].second.y, segments[i].second.x),
                { }, { &segments[i] }); // end event
  }

  Segment* left;
  Segment* right;
  int i = 0;
  while(sl.nextEvent() && i<20)
  {
    std::cout << i++ << std::endl;
  }

  return 0;
}

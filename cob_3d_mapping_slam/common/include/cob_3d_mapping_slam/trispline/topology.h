/*
 * trispline.h
 *
 *  Created on: 25.10.2012
 *      Author: josh
 */

#ifndef TOPOLGY_H_
#define TOPOLGY_H_

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Cartesian.h>
#include <CGAL/Homogeneous_d.h>
#include <CGAL/leda_integer.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <unsupported/Eigen/NonLinearOptimization>
#include <eigen3/Eigen/Jacobi>

#include <cob_3d_mapping_slam/marker/marker_container.h>

namespace ParametricSurface {

  class Topology
  {
    //typedef leda_integer RT;
    typedef float RT;
    //typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef CGAL::Cartesian<double>                               Kernel;
    //typedef CGAL::Homogeneous_d<double>                               Kernel;
    typedef CGAL::Delaunay_triangulation_2<Kernel> Delaunay_d;
    typedef Delaunay_d::Point Point;
    typedef Kernel::Vector_2 Vector;
    typedef Delaunay_d::Vertex Vertex;
    typedef Delaunay_d::Face Face;
    typedef Delaunay_d::Face_handle Face_handle;
    typedef Delaunay_d::Face_iterator Face_iterator;

    //typedef Delaunay_d::Simplex_handle Simplex_handle;
    //typedef Delaunay_d::Simplex_const_iterator Simplex_const_iterator;
    //typedef Delaunay_d::Point_const_iterator Point_const_iterator;
    //typedef Delaunay_d::Simplex_iterator Simplex_iterator;
    typedef Delaunay_d::Vertex_handle Vertex_handle;
    typedef Delaunay_d::Face_circulator Face_circulator;

  public:

    struct  POINT {
      Eigen::Vector2f uv;
      Eigen::Vector3f pt, n, n2;
      Vertex_handle vh;
      float weight_;

      void transform(const Eigen::Matrix3f &rot, const Eigen::Vector3f &tr) {
        pt = rot*pt + tr;
        n  = rot*n;
        n2 = rot*n2;
      }
    };

  private:

    Delaunay_d del_;
    std::vector< boost::shared_ptr<POINT> > pts_;
    std::vector< boost::shared_ptr<ParametricSurface::TriSpline2_Fade> > tris_;
    std::map< Vertex*, boost::shared_ptr<POINT> > map_pts_;
    std::map< Face*, boost::shared_ptr<ParametricSurface::TriSpline2_Fade> > map_tris_;
    float thr_;


    bool canRemove(const boost::shared_ptr<POINT> &pt) {
      //1. if its part of border -> don't delete (for now)
      Face_circulator fc = pt->vh->incident_faces();
      if (fc != 0) {
        int n=0;
        bool b1=false, b2=false;
        Vector p;
        do {
          Vertex_handle v1 = fc->vertex(0), v2 = fc->vertex(1);
          if(v1==pt->vh) v1 = fc->vertex(2);
          if(v2==pt->vh) v2 = fc->vertex(2);

          std::cout<<"p1\n"<<v1->point()<<"\n";
          std::cout<<"p2\n"<<v2->point()<<"\n";

          if(!n) {
            p = v1->point()-pt->vh->point();
          }
          float f;
          Vector v;

          v = v1->point()-pt->vh->point();
          if( v.x()*p.x()+v.y()*p.y()<=0 ) {
            f = v.x()*p.y()-v.y()*p.x();
            if(f>0) b1=true; else if(f<0) b2=true;

            std::cout<<"b "<<b1<<" "<<b2<<"\n";
            std::cout<<"f2 "<<f<<"\n";
          }

          v = v2->point()-pt->vh->point();
          if( v.x()*p.x()+v.y()*p.y()<=0 ) {
            f = v.x()*p.y()-v.y()*p.x();
            if(f>0) b1=true; else if(f<0) b2=true;

            std::cout<<"b "<<b1<<" "<<b2<<"\n";
            std::cout<<"f2 "<<f<<"\n";
          }

          if(b1&&b2) break;
          ++n;
        } while (++fc != pt->vh->incident_faces());

        if(!b1||!b2) return false;
      }
      else
        return false;

      //2. get PCA of \delta of each edge connecting vertex
      //   -> if smallest eigen value less than threshold
      //   -> delete
      Eigen::Matrix3f V = Eigen::Matrix3f::Zero();

      fc = pt->vh->incident_faces();
      float n=0;
      do {
        ROS_ASSERT(map_tris_.find(&*fc)!=map_tris_.end());

        Eigen::Matrix3f delta = map_tris_.find(&*fc)->second->delta();
        for(int i=0; i<3; i++)
          V+=delta.col(i)*delta.col(i).transpose();
        n+=3.f;
      } while (++fc != pt->vh->incident_faces());

      Eigen::JacobiSVD<Eigen::Matrix3f> svd (V, Eigen::ComputeFullU | Eigen::ComputeFullV);
      return
          std::min(std::min(std::abs(svd.singularValues()(0)),std::abs(svd.singularValues()(1))),std::abs(svd.singularValues()(2)))
      <thr_*n;
    }

    void removePoint(const size_t i) {
      map_pts_.erase(&*pts_[i]->vh);
      del_.remove(pts_[i]->vh);
      pts_.erase(pts_.begin()+i);
    }

    // for LM
    struct Functor {
      Eigen::Vector3f pt_;
      const Topology &top_;

      Functor(const Topology &t, const Eigen::Vector3f &pt):pt_(pt), top_(t)
      {}

      int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
      {
        // distance
        fvec(0) = (pt_ - top_.project2world(x)).squaredNorm();
        fvec(1) = 0;

        std::cout<<"x\n"<<x<<"\n";
        std::cout<<"dist "<<fvec(0)<<"\n";
        return 0;
      }

      int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
      {
        //Jacobian
#if 0
        Eigen::Matrix4f p = top_.normalAtUV(x);

        Eigen::Vector3f vx, vy;
        vx = p.col(3).head<3>().cross(p.col(1).head<3>()).cross(p.col(3).head<3>());
        vy = p.col(3).head<3>().cross(p.col(2).head<3>()).cross(p.col(3).head<3>());

        vx.normalize();
        vy.normalize();

        Eigen::Matrix3f M;
        M.col(0) = vx;
        M.col(1) = -vy;
        M.col(2) = p.col(3).head<3>();

        fjac.row(0) = 2*(M.inverse()*(p.col(0).head<3>()-pt_)).head<2>();

        std::cout<<"o1 "<<vx.dot(p.col(0).head<3>()-pt_)<<"\n";
        std::cout<<"o2 "<<vx.dot(p.col(1).head<3>()-pt_)<<"\n";
        std::cout<<"p\n"<<p<<"\n";
#else
        Eigen::Vector3f v = top_.project2world(x);
        Eigen::Matrix<float,3,2> M = top_.normal(x);

        fjac(0,0) = 2*M.col(0).dot( v-pt_ );
        fjac(0,1) = 2*M.col(1).dot( v-pt_ );
#endif

        //        fjac(0,0) = 2*( p.col(1)(3)* p.col(3).head<3>().dot(p.col(0).head<3>()-pt_) + vx.dot(p.col(0).head<3>()-pt_));
        //        fjac(0,1) = -2*( p.col(2)(3)* p.col(3).head<3>().dot(p.col(0).head<3>()-pt_) + vy.dot(p.col(0).head<3>()-pt_));
        fjac(1,0) = fjac(1,1) = 0;

        std::cout<<"fjac\n"<<fjac<<"\n";
        std::cout<<"d\n"<<M<<"\n";

        return 0;
      }

      int inputs() const { return 2; }
      int values() const { return 2; } // number of constraints
    };

    static float sqDistLinePt(const Eigen::Vector2f &uv, const Eigen::Vector2f &r1, const Eigen::Vector2f &r2) {
      float f = (uv-r1).dot(r2-r1) / (r2-r1).dot(r2-r1);
      if(f>1) f=1;
      else if(f<0) f=0;
      return (uv - f*(r2-r1) - r1).squaredNorm();
    }

    inline ParametricSurface::TriSpline2_Fade* locate(const Eigen::Vector2f &uv) const {
      bool inside;
      return locate(uv,inside);
    }
    inline ParametricSurface::TriSpline2_Fade* locate(const Eigen::Vector2f &uv, bool &inside) const {
      Face_handle sh = del_.locate( Point(uv(0),uv(1)) );

      if(map_tris_.find(&*sh)==map_tris_.end()) {
        ParametricSurface::TriSpline2_Fade* r=NULL;
        inside=false;

        float mi = std::numeric_limits<float>::max();
        for(std::map< Face*, boost::shared_ptr<ParametricSurface::TriSpline2_Fade> >::const_iterator it = map_tris_.begin();
            it!=map_tris_.end(); ++it)
        {

          Eigen::Vector2f p1 = it->second->getUV(0),p2=it->second->getUV(1),p3=it->second->getUV(2);

          const float dist = std::min( sqDistLinePt(uv, p1,p2), std::min(sqDistLinePt(uv, p2,p3), sqDistLinePt(uv, p3,p1)));

          std::cout<<"dist "<<dist<<"\n";
          if(dist<mi) {
            mi = dist;
            r = it->second.get();
          }

        }

        ROS_ASSERT(r);
        return r;
      }
      else {
        inside=true;
        return (ParametricSurface::TriSpline2_Fade*)map_tris_.find(&*sh)->second.get();
      }

    }

  public:

    Topology(const float thr): thr_(thr)
    {
    }

    void insertPoint(const POINT &pt) {
      insertPointWithoutUpdate(pt);
      update();
    }

    void finish() {
      for(size_t i=0; i<pts_.size(); i++) {
        if( canRemove(pts_[i]) ) {
          std::cout<<"remove point\n";
          removePoint(i);
          update();
          --i;
        }
      }

    }

    void insertPointWithoutUpdate(const POINT &pt) {
      ROS_INFO("insert pt");
      pts_.push_back(boost::shared_ptr<POINT>( new POINT(pt)));
      pts_.back()->vh = del_.insert(Point(pt.uv(0),pt.uv(1)));
      pts_.back()->weight_ = 1.f; //TODO:
      if(map_pts_.find(&*pts_.back()->vh)!=map_pts_.end()) {
        ROS_WARN("pt (%f, %f) already there", pt.uv(0), pt.uv(1));
        pts_.erase(pts_.end()-1);
        return;
      }
      //pts_.back()->vh->pp = pts_.back().get();
      map_pts_[&*pts_.back()->vh] = pts_.back();
    }

    void update() {

      tris_.clear();
      map_tris_.clear();
      if(pts_.size()>2) {
        for(Face_iterator it = del_.faces_begin(); it!=del_.faces_end(); it++) {
          POINT *a = map_pts_[&*it->vertex(0)].get();
          POINT *b = map_pts_[&*it->vertex(1)].get();
          POINT *c = map_pts_[&*it->vertex(2)].get();
          tris_.push_back( boost::shared_ptr<ParametricSurface::TriSpline2_Fade>(new ParametricSurface::TriSpline2_Fade(
              a->pt,      b->pt,  c->pt,
              a->n,       b->n,   c->n,
              a->n2,      b->n2,  c->n2,
              a->uv,      b->uv,  c->uv
          ) ));

          //it->pp = tris_.back().get();
          map_tris_[&*it] = tris_.back();
        }
      }

    }

    inline Eigen::Vector2f nextPoint(const Eigen::Vector3f &p) const {
      //std::cout<<"nP\n"<<p<<"\n";
      Eigen::VectorXf r(2);

      float dis = std::numeric_limits<float>::max();
      for(std::map< Face*, boost::shared_ptr<ParametricSurface::TriSpline2_Fade> >::const_iterator it = map_tris_.begin();
          it!=map_tris_.end(); ++it)
      {
        const float d = (it->second->getMid()-p).squaredNorm();
        if(d<dis) {
          dis = d;
          r = it->second->getMidUV();
        }
      }
      Functor functor(*this, p);
      Eigen::LevenbergMarquardt<Functor, float> lm(functor);
      lm.parameters.maxfev = 50; //not to often (performance)
      lm.minimize(r);

      return r;
    }

    inline Eigen::Vector3f project2world(const Eigen::Vector2f &uv) const {
      return (*locate(uv))(uv);
    }

    inline Eigen::Vector3f normalAt(const Eigen::Vector2f &uv) const {
      Eigen::Vector3f v = locate(uv)->normalAt(uv);
      v.normalize();
      return v;
    }

    inline Eigen::Matrix<float,3,2> normal(const Eigen::Vector2f &uv) const {
      return locate(uv)->normal(uv);
    }

    inline Eigen::Vector3f normalAt2(const Eigen::Vector2f &uv, bool &inside) const {
      return locate(uv, inside)->normalAt2(uv);
    }

    /*inline Eigen::Matrix4f normalAtUV(const Eigen::Vector2f &uv) const {
      return locate(uv)->normalAtUV(uv);
    }*/

    void operator+=(const Topology &o) {
      ROS_INFO("merge");
      if(pts_.size()<3)
      {
        ROS_INFO("copy %d", pts_.size());

        for(size_t i=0; i<o.pts_.size(); i++) {
          insertPointWithoutUpdate(*o.pts_[i]);
        }
      }
      else {
        ROS_INFO("add %d",o.pts_.size());

        std::vector<POINT> temp;
        for(size_t i=0; i<o.pts_.size(); i++) {
          POINT p = *o.pts_[i];
          p.uv = nextPoint(p.pt);
          bool inside;
          Eigen::Vector3f t = normalAt2(p.uv,inside);
          std::cout<<"normal2 "<<t<<"\n";
          if(inside) {
            float w=1;
            p.n2 = (p.weight_*p.n2 + w*t)/(p.weight_+w);

            std::cout<<"w "<<p.weight_<<"\n";
            std::cout<<"n bef\n"<<p.n<<"\n";

            t = normalAt(p.uv);
            p.n = (p.weight_*p.n + w*t)/(p.weight_+w);
            p.n.normalize();

            std::cout<<"n after\n"<<p.n<<"\n";

            t = project2world(p.uv);
            p.pt = (p.weight_*p.pt + w*t)/(p.weight_+w);
          }
          temp.push_back(p);
        }
        for(size_t i=0; i<pts_.size(); i++) {
          Eigen::Vector2f uv = o.nextPoint(pts_[i]->pt);
          bool inside;
          Eigen::Vector3f t = o.normalAt2(uv,inside);
          if(inside) {
            float w=1;
            pts_[i]->n2 = (pts_[i]->weight_*pts_[i]->n2 + w*t)/(pts_[i]->weight_+w);

            t = o.normalAt(uv);
            pts_[i]->n = (pts_[i]->weight_*pts_[i]->n + w*t)/(pts_[i]->weight_+w);
            pts_[i]->n.normalize();

            t = o.project2world(uv);
            pts_[i]->pt = (pts_[i]->weight_*pts_[i]->pt + w*t)/(pts_[i]->weight_+w);
          }
        }
        for(size_t i=0; i<temp.size(); i++) insertPointWithoutUpdate(temp[i]);
      }
      update();

      //finish();
    }

    void transform(const Eigen::Matrix3f &rot, const Eigen::Vector3f &tr) {
      for(size_t i=0; i<pts_.size(); i++) {
        pts_[i]->transform(rot, tr);
      }

      for(std::map< Face*, boost::shared_ptr<ParametricSurface::TriSpline2_Fade> >::iterator it = map_tris_.begin();
          it!=map_tris_.end(); ++it)
      {
        it->second->transform(rot, tr);
      }
    }

    void print() const {
      for(size_t i=0; i<pts_.size(); i++) {
        std::cout<<"uv\n"<<pts_[i]->uv<<"\n";
        std::cout<<"pt\n"<<pts_[i]->pt<<"\n";
      }
    }

    void add(cob_3d_marker::MarkerList_Line &ml) const {
      for(std::map< Face*, boost::shared_ptr<ParametricSurface::TriSpline2_Fade> >::const_iterator it = map_tris_.begin();
          it!=map_tris_.end(); ++it)
      {
        ParametricSurface::_Line l[6];
        it->second->test_setup(l);
        for(int i=0; i<6; i++)
          ml.addLine( l[i].o, l[i].o+l[i].u, 0,1,0);
        for(int i=0; i<3; i++) {
          ml.addLine( it->second->getEdge(i), it->second->getEdge((i+1)%3) );

          ml.addLine( it->second->getEdge(i), it->second->getFade(i)(1,0) , 0.8f,0.1f,0.1f);
          ml.addLine( it->second->getFade(i)(1,0), it->second->getFade(i)(0,1) , 0.8f,0.1f,0.1f);
          ml.addLine( it->second->getFade(i)(0,1), it->second->getEdge((i+1)%3) , 0.8f,0.1f,0.1f);
        }
      }
    }

    void add(cob_3d_marker::MarkerList_Triangles &mt) const {
      for(std::map< Face*, boost::shared_ptr<ParametricSurface::TriSpline2_Fade> >::const_iterator it = map_tris_.begin();
          it!=map_tris_.end(); ++it)
      {
        /*for(int i=0; i<5; i++) {
          mt.addTriangle( it->second->getEdge(i), it->second->getEdge((i+1)%3) );

          ml.addLine( it->second->getEdge(i), it->second->getFade(i)(1,0) , 0.8f,0.1f,0.1f);
          ml.addLine( it->second->getFade(i)(1,0), it->second->getFade(i)(0,1) , 0.8f,0.1f,0.1f);
          ml.addLine( it->second->getFade(i)(0,1), it->second->getEdge((i+1)%3) , 0.8f,0.1f,0.1f);
        }*/
      }
    }

    void add(cob_3d_marker::MarkerList_Arrow &ma) const {
      int j=0;
      for(std::map< Face*, boost::shared_ptr<ParametricSurface::TriSpline2_Fade> >::const_iterator it = map_tris_.begin();
          it!=map_tris_.end(); ++it)
      {
        ++j;
        for(int i=0; i<3; i++) {
          ma.addArrow(it->second->getEdge(i), it->second->getEdge(i)+it->second->getNormal(i));

          ma.addArrow(it->second->getEdge(i), it->second->getEdge(i)+it->second->getNormal2(i), 1,0,0);
        }
      }
    }

  };

}

#endif /* TOPOLGY_H_ */

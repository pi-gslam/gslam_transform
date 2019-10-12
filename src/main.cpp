#include <TooN/sim3.h>

#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"

#include <GSLAM/core/GSLAM.h>

#ifdef HAS_CERES
#include <ceres/rotation.h>
#endif

using namespace std;

void gslam_time(){
    int num=svar.GetInt("num",1e6);
    std::vector<GSLAM::SO3d>  rs;
    std::vector<GSLAM::SE3d>  ts;
    std::vector<GSLAM::SIM3d> ss;
    std::vector<GSLAM::Point3d> ps;


    default_random_engine e;

    uniform_real_distribution<double> rtp(-3.15,3.15);

    for(int i=0;i<num;i++){
        GSLAM::Point3d r(rtp(e),rtp(e),rtp(e));
        GSLAM::Point3d t(rtp(e),rtp(e),rtp(e));
        GSLAM::Point3d p(rtp(e),rtp(e),rtp(e));
        double s=uniform_real_distribution<double>(1.,2.)(e);
        rs.push_back(GSLAM::SO3d::exp(r));
        ts.push_back(GSLAM::SE3d(rs[i],t));
        ss.push_back(GSLAM::SIM3d(ts[i],s));
        ps.push_back(p);
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SO3::mult");
        for(auto& r:rs) r=r*r;
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SO3::trans");
        for(int i=0;i<rs.size();i++) ps[i]=rs[i]*ps[i];
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SO3::exp");
        for(int i=0;i<rs.size();i++) rs[i]=GSLAM::SO3d::exp(ps[i]);
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SO3::ln");
        for(int i=0;i<rs.size();i++) ps[i]=rs[i].log();
    }

    {
        GSLAM::ScopedTimer tm("GSLAM::SE3::mult");
        for(auto& t:ts) t=t*t;
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SE3::trans");
        for(int i=0;i<ts.size();i++) ps[i]=ts[i]*ps[i];
    }

    std::vector<GSLAM::Vector6d > se3_algebra(num);

    {
        GSLAM::ScopedTimer t("GSLAM::SE3::ln");
        for(int i=0;i<ts.size();i++) se3_algebra[i]=ts[i].log();
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SE3::exp");
        for(int i=0;i<num;i++) ts[i]=GSLAM::SE3d::exp(se3_algebra[i]);
    }

    {
        GSLAM::ScopedTimer tm("GSLAM::SIM3::mult");
        for(auto& t:ss) t=t*t;
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SIM3::trans");
        for(int i=0;i<num;i++) ps[i]=ss[i]*ps[i];
    }

    std::vector<GSLAM::Vector<double,7>> sim3_algebra(num);

    {
        GSLAM::ScopedTimer t("GSLAM::SIM3::ln");
        for(int i=0;i<ts.size();i++) se3_algebra[i]=ts[i].log();
    }

    {
        GSLAM::ScopedTimer t("GSLAM::SIM3::exp");
        for(int i=0;i<num;i++) ts[i]=GSLAM::SE3d::exp(se3_algebra[i]);
    }

}

void sophus_time()
{
    int num=svar.GetInt("num",1e6);
    std::vector<Sophus::SO3d>  rs;
    std::vector<Sophus::SE3d>  ts;
    std::vector<Sophus::Sim3d>  ss;
    std::vector<Eigen::Vector3d> ps;

    default_random_engine e;
    uniform_real_distribution<double> rtp(-3.15,3.15);

    for(int i=0;i<num;i++){
        GSLAM::Point3d r(rtp(e),rtp(e),rtp(e));
        GSLAM::Point3d t(rtp(e),rtp(e),rtp(e));
        GSLAM::Point3d p(rtp(e),rtp(e),rtp(e));
        double s=uniform_real_distribution<double>(1.,2.)(e);
        rs.push_back(GSLAM::SO3d::exp(r));
        ts.push_back(GSLAM::SE3d(rs[i],t));
        ss.push_back(GSLAM::SIM3d(ts[i],s));
        ps.push_back(p);
    }

    {
        GSLAM::ScopedTimer t("Sophus::SO3::mult");
        for(auto& r:rs) r=r*r;
    }

    {
        GSLAM::ScopedTimer t("Sophus::SO3::trans");
        for(int i=0;i<rs.size();i++) ps[i]=rs[i]*ps[i];
    }

    {
        GSLAM::ScopedTimer t("Sophus::SO3::exp");
        for(int i=0;i<rs.size();i++) rs[i]=Sophus::SO3d::exp(ps[i]);
    }

    {
        GSLAM::ScopedTimer t("Sophus::SO3::ln");
        for(int i=0;i<rs.size();i++) ps[i]=rs[i].log();
    }

    {
        GSLAM::ScopedTimer tm("Sophus::SE3::mult");
        for(auto& t:ts) t=t*t;
    }

    {
        GSLAM::ScopedTimer t("Sophus::SE3::trans");
        for(int i=0;i<ts.size();i++) ps[i]=ts[i]*ps[i];
    }

    std::vector<Eigen::Matrix<double,6,1> > se3_algebra(num);

    {
        GSLAM::ScopedTimer t("Sophus::SE3::ln");
        for(int i=0;i<ts.size();i++) se3_algebra[i]=ts[i].log();
    }

    {
        GSLAM::ScopedTimer t("Sophus::SE3::exp");
        for(int i=0;i<num;i++) ts[i]=Sophus::SE3d::exp(se3_algebra[i]);
    }

    {
        GSLAM::ScopedTimer tm("Sophus::SIM3::mult");
        for(auto& t:ss) t=t*t;
    }

    {
        GSLAM::ScopedTimer t("Sophus::SIM3::trans");
        for(int i=0;i<num;i++) ps[i]=ss[i]*ps[i];
    }

    std::vector<Eigen::Matrix<double,7,1> > sim3_algebra(num);

    {
        GSLAM::ScopedTimer t("Sophus::SIM3::ln");
        for(int i=0;i<ss.size();i++) sim3_algebra[i]=ss[i].log();
    }

    {
        GSLAM::ScopedTimer t("Sophus::SIM3::exp");
        for(int i=0;i<num;i++) ss[i]=Sophus::Sim3d::exp(sim3_algebra[i]);
    }

}

void toon_time()
{
    typedef TooN::SO3<double>  SO3d;
    typedef TooN::SE3<double>  SE3d;
    typedef TooN::SIM3<double> SIM3d;
    typedef TooN::Vector<3,double> Vector3d;
    int num=svar.GetInt("num",1e6);
    std::vector<SO3d >    rs;
    std::vector<SE3d>     ts;
    std::vector<SIM3d>    ss;
    std::vector<Vector3d> ps;


    default_random_engine e;
    uniform_real_distribution<double> rtp(-3.15,3.15);

    for(int i=0;i<num;i++){
        GSLAM::Point3d r(rtp(e),rtp(e),rtp(e));
        GSLAM::Point3d t(rtp(e),rtp(e),rtp(e));
        GSLAM::Point3d p(rtp(e),rtp(e),rtp(e));
        double s=uniform_real_distribution<double>(1.,2.)(e);
        SO3d R;
        GSLAM::SO3d::exp(r).getMatrix((double*)&R);
        rs.push_back(R);
        ts.push_back(SE3d(R,Vector3d(t)));
        ss.push_back(SIM3d(R,Vector3d(t),s));
        ps.push_back(p);
    }

    {
        GSLAM::ScopedTimer t("TooN::SO3::mult");
        for(auto& r:rs) r=r*r;
    }

    {
        GSLAM::ScopedTimer t("TooN::SO3::trans");
        for(int i=0;i<rs.size();i++) ps[i]=rs[i]*ps[i];
    }

    {
        GSLAM::ScopedTimer t("TooN::SO3::exp");
        for(int i=0;i<rs.size();i++) rs[i]=SO3d::exp(ps[i]);
    }

    {
        GSLAM::ScopedTimer t("TooN::SO3::ln");
        for(int i=0;i<rs.size();i++) ps[i]=rs[i].ln();
    }

    {
        GSLAM::ScopedTimer tm("TooN::SE3::mult");
        for(auto& t:ts) t=t*t;
    }

    {
        GSLAM::ScopedTimer t("TooN::SE3::trans");
        for(int i=0;i<ts.size();i++) ps[i]=ts[i]*ps[i];
    }

    std::vector<TooN::Vector<6,double> > se3_algebra(num);

    {
        GSLAM::ScopedTimer t("TooN::SE3::ln");
        for(int i=0;i<ts.size();i++) se3_algebra[i]=ts[i].ln();
    }

    {
        GSLAM::ScopedTimer t("TooN::SE3::exp");
        for(int i=0;i<num;i++) ts[i]=SE3d::exp(se3_algebra[i]);
    }

    {
        GSLAM::ScopedTimer tm("TooN::SIM3::mult");
        for(auto& t:ss) t=t*t;
    }

    {
        GSLAM::ScopedTimer t("TooN::SIM3::trans");
        for(int i=0;i<num;i++) ps[i]=ss[i]*ps[i];
    }

    std::vector<TooN::Vector<7,double> > sim3_algebra(num);

    {
        GSLAM::ScopedTimer t("TooN::SIM3::ln");
        for(int i=0;i<ss.size();i++) sim3_algebra[i]=ss[i].ln();
    }

    {
        GSLAM::ScopedTimer t("TooN::SIM3::exp");
        for(int i=0;i<num;i++) ss[i]=SIM3d::exp(sim3_algebra[i]);
    }

}

void ceres_time(){
    int num=svar.GetInt("num",1e6);
    std::vector<Eigen::Vector3d>            rs;
    std::vector<Eigen::Matrix<double,6,1> > ts;
    std::vector<Eigen::Matrix<double,7,1> > ss;
    std::vector<Eigen::Vector3d>           ps;

    default_random_engine e;
    uniform_real_distribution<double> rtp(-3.15,3.15);
    for(int i=0;i<num;i++){
        Eigen::Matrix<double,7,1> s;
        s<<rtp(e),rtp(e),rtp(e),rtp(e),rtp(e),rtp(e),rtp(e);
        Eigen::Vector3d p(rtp(e),rtp(e),rtp(e));
        rs.push_back(s.block(0,0,3,1));
        ts.push_back(s.block(0,0,6,1));
        ss.push_back(s);
        ps.push_back(p);
    }

    {
        GSLAM::ScopedTimer t("Ceres::SO3::mult");
        for(auto& r:rs) {
            double q1[4],q2[4];
            ceres::AngleAxisToQuaternion(&r[0],q1);
            ceres::AngleAxisToQuaternion(&r[0],q2);
            ceres::QuaternionProduct(q1,q2,&r[0]);
        }
    }

    {
        GSLAM::ScopedTimer t("Ceres::SO3::trans");
        for(int i=0;i<rs.size();i++)
            ceres::AngleAxisRotatePoint(&rs[i][0],&ps[i][0],&ps[i][0]);
    }

//    {
//        GSLAM::ScopedTimer t("Ceres::SO3::exp");
//        for(int i=0;i<rs.size();i++) rs[i]=GSLAM::SO3d::exp(ps[i]);
//    }

//    {
//        GSLAM::ScopedTimer t("Ceres::SO3::ln");
//        for(int i=0;i<rs.size();i++) ps[i]=rs[i].log();
//    }

//    {
//        GSLAM::ScopedTimer tm("Ceres::SE3::mult");
//        for(auto& t:ts) t=t*t;
//    }

//    {
//        GSLAM::ScopedTimer t("Ceres::SE3::trans");
//        for(int i=0;i<ts.size();i++) ps[i]=ts[i]*ps[i];
//    }

//    std::vector<GSLAM::Array_<double,6> > se3_algebra(num);

//    {
//        GSLAM::ScopedTimer t("Ceres::SE3::ln");
//        for(int i=0;i<ts.size();i++) se3_algebra[i]=ts[i].log();
//    }

//    {
//        GSLAM::ScopedTimer t("Ceres::SE3::exp");
//        for(int i=0;i<num;i++) ts[i]=GSLAM::SE3d::exp(se3_algebra[i]);
//    }

//    {
//        GSLAM::ScopedTimer tm("Ceres::SIM3::mult");
//        for(auto& t:ss) t=t*t;
//    }

//    {
//        GSLAM::ScopedTimer t("Ceres::SIM3::trans");
//        for(int i=0;i<num;i++) ps[i]=ss[i]*ps[i];
//    }

//    std::vector<GSLAM::Array_<double,7> > sim3_algebra(num);

//    {
//        GSLAM::ScopedTimer t("Ceres::SIM3::ln");
//        for(int i=0;i<ts.size();i++) se3_algebra[i]=ts[i].log();
//    }

//    {
//        GSLAM::ScopedTimer t("Ceres::SIM3::exp");
//        for(int i=0;i<num;i++) ts[i]=GSLAM::SE3d::exp(se3_algebra[i]);
//    }


}

void check()
{
    int& checknum=svar.GetInt("check_num",100);
    if(checknum<=0) return;
    checknum--;
}

int run_transform(GSLAM::Svar config){
    config.arg<int>("num",1000000,"Number of times to repeat the computation.");
    config.arg<int>("check_num",100,"Number of times to repeat the check.");
    config.arg<bool>("check",true,"Check the result and report");
    svar=config;

    if(config.get("help",false)) return config.help();

    gslam_time();
    sophus_time();
    toon_time();
    ceres_time();
    if(svar.Get<bool>("check")){
        check();
    }
    return 0;
}

GSLAM_REGISTER_APPLICATION(transform,run_transform);

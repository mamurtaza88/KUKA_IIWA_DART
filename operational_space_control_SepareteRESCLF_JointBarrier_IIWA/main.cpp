/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include <dart/dart.hpp>
#include <dart/gui/osg/osg.hpp>
#include <dart/utils/urdf/urdf.hpp>
#include <dart/utils/utils.hpp>

#include <nlopt.hpp>
#include <cmath>
#include <iostream>
#include <fstream>

#include <dart/gui/osg/InteractiveFrame.hpp>

using namespace dart::common;
using namespace dart::dynamics;
using namespace dart::math;
using namespace std;

using namespace dart::gui::osg;

// *************************** Read Matrix to read from text file **************************
#define MAXBUFSIZE  ((int) 1e6)
Eigen::MatrixXd readMatrix(const char *filename)
{
  int cols = 0, rows = 0;
  double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(filename);
  while (! infile.eof())
  {
    std::string line;
    getline(infile, line);

    int temp_cols = 0;
    std::stringstream stream(line);
    while(! stream.eof())
      stream >> buff[cols*rows+temp_cols++];

      if (temp_cols == 0)
        continue;

      if (cols == 0)
        cols = temp_cols;

        rows++;
  }

  infile.close();

  rows--;

    // Populate matrix with numbers.
  Eigen::MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
     for (int j = 0; j < cols; j++)
        result(i,j) = buff[ cols*i+j ];

  return result;
};

//   ************************ Adding Barrier Function Optimization Stuff *******************

struct OptParams_CBF {
  Eigen::MatrixXd L_G;
  Eigen::VectorXd L_F;
  Eigen::MatrixXd K;
  Eigen::VectorXd eta;
  Eigen::VectorXd u_nom;
};

double optFunc_CBF(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data) {
  OptParams_CBF* optParams_CBF = reinterpret_cast<OptParams_CBF*>(my_func_data);
  // std::cout << "done reading optParams in optFunc_CBF" << std::endl;

  size_t n = x.size();
  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) X(i) = x[i];
  // std::cout << "done reading x" << std::endl;

  if (!grad.empty()) {
    Eigen::MatrixXd mGrad = X - optParams_CBF->u_nom;
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
  }
 
   // std::cout << "Returning output" << std::endl;
  double check1 = 0.5 * pow((X - optParams_CBF->u_nom).norm(), 2);
   // std::cout << "about to return something from optFunc_CBF" << std::endl;
  return check1;
}

double constraintFunc_CBF(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  OptParams_CBF* constParams = reinterpret_cast<OptParams_CBF*>(my_func_data);
  // std::cout << "done reading optParams in constraintFunc_RESCLF1" << std::endl;
  
  size_t n = x.size();
  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) X(i) = x[i];

  if (!grad.empty()) {
    Eigen::MatrixXd mGrad = Eigen::VectorXd::Zero(n);
  // Gradient has to be negative since barrier is positive as per the formulation    
    for(int i = 0;i < n ;i++){
      mGrad(i,0) = -constParams->L_G(0,i);
    }
    
    // std::cout << "constraintFunc_CBF Check 2" << std::endl;
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
  }

  Eigen::Matrix<double,1 ,1> mResult;
  mResult = constParams->L_G * X + constParams->L_F + constParams->K * constParams->eta;
  // std::cout << "constraintFunc_CBF Check 7" << std::endl;
  double result;

  //Send the negative of the result computed since the formulation is positive
  result = -mResult(0,0);
  // std::cout << "Returning result" << std::endl;
  return result;

}


//********************************************RES-CLF Optimization Parameters
struct OptParams{
  Eigen::MatrixXd P;
  Eigen::VectorXd b;
};

struct OptParams_RESCLF {
  Eigen::MatrixXd L_G_pos;
  Eigen::VectorXd L_F_pos;
  Eigen::VectorXd V_x_pos;
  Eigen::MatrixXd L_G_ori;
  Eigen::VectorXd L_F_ori;
  Eigen::VectorXd V_x_ori;
  double gamma;
  double relaxation;
};


double optFunc_RESCLF(const std::vector<double>& x, std::vector<double>& grad,
               void* my_func_data) {
  OptParams_RESCLF* optParams_RESCLF = reinterpret_cast<OptParams_RESCLF*>(my_func_data);
  // std::cout << "Starting reading optParams in optFunc_RESCLF" << std::endl;
  // Eigen::Matrix<double, 18, 1> X(x.data());
  size_t n = x.size();
  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) X(i) = x[i];
  // std::cout << "done reading x" << std::endl;

  if (!grad.empty()) {
    Eigen::MatrixXd mGrad = 2 * X;
    mGrad(0,0) = 2*optParams_RESCLF->relaxation*X(0);
    // std::cout << "done calculating gradient in optFunc_RESCLF" << std::endl;
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
    // std::cout << "done changing gradient cast in optFunc_RESCLF" << std::endl;
  }
  // std::cout << "about to return something from optFunc_RESCLF" << std::endl;
  double output = 0;
    for(int i = 1; i < n;i++){
      output = output + pow(X(i),2);
    }
    output = output + optParams_RESCLF->relaxation*pow(X(0),2);
    // std::cout << "Returning output of optFunc_RESCLF" << std::endl;
  return output;
}

double constraintFunc_RESCLF1(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  OptParams_RESCLF* constParams = reinterpret_cast<OptParams_RESCLF*>(my_func_data);
  
  // std::cout << "Starting reading optParams in constraintFunc_RESCLF1" << std::endl;
  // std::cout << "size of constParams->L_G = " << constParams->L_G.rows() << "*" << constParams->L_G.cols() << std::endl;
  // double gamma = optParams->gamma;
  size_t n = x.size();
  // Eigen::Matrix<double, 8, 1> X(x.data());
  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) X(i) = x[i];

  if (!grad.empty()) {
    Eigen::MatrixXd mGrad = Eigen::VectorXd::Zero(n);
    
    for(int i = 1;i < 4;i++){
      mGrad(i,0) = constParams->L_G_pos(0,i-1);
    }
    
    mGrad(0,0) = -1;
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
    
  }
  //   // if (!grad.empty()) {
    //  grad[0] = 1;
    //  grad[1] = 0;
    // }
  // std::cout << "Computing mResult" << std::endl;
  Eigen::Matrix<double,1 ,1> mResult;

  //Match dimension here. X.segment(1,3) is used since we are considering position only. It means vector of 
  // block 3 starting at index 1. Index 0 is the relaxation term. 

  mResult = constParams->L_G_pos * X.segment(1,3) + constParams->L_F_pos + constParams->gamma * constParams->V_x_pos - X.segment<1>(0);//constParams->relaxation*Eigen::VectorXd::Ones(1);
  // mResult = mResult.col(0) - X(0);
  
  double result;
  result = mResult(0,0);
  // std::cout << "Returning output of constraintFunc_RESCLF1" << std::endl;

  return result;

}

double constraintFunc_RESCLF2(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  
  OptParams_RESCLF* constParams = reinterpret_cast<OptParams_RESCLF*>(my_func_data);
  // std::cout << "Starting reading optParams in constraintFunc_RESCLF2" << std::endl;
  // std::cout << "size of constParams->L_G_ori = " << constParams->L_G_ori.rows() << "*" << constParams->L_G_ori.cols() << std::endl;
  // double gamma = optParams->gamma;
  size_t n = x.size();
  // Eigen::Matrix<double, 8, 1> X(x.data());
  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) X(i) = x[i];

  if (!grad.empty()) {
    Eigen::MatrixXd mGrad = Eigen::VectorXd::Zero(n);
    
    for(int i = 4;i < 7;i++){
      mGrad(i,0) = constParams->L_G_ori(0,i-4); // Since index starts at 4 and L_G_ori starts at zero index. Hence fix it.
    }
    
    // No relaxation term in orientation. Add if required.
    // mGrad(0,0) = -1; 
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
    
  }
  //   // if (!grad.empty()) {
    //  grad[0] = 1;
    //  grad[1] = 0;
    // }
  // std::cout << "Computing mResult in constraintFunc_RESCLF2" << std::endl;
  Eigen::Matrix<double,1 ,1> mResult;

  //Match dimension here. X.segment(4,3) is used since we are considering orientation only. It means vector of 
  // block 3 starting at index 4. Index 0 is the relaxation term. 

  mResult = constParams->L_G_ori * X.segment(4,3) + constParams->L_F_ori + constParams->gamma * constParams->V_x_ori;
  // mResult = mResult.col(0) - X(0);
  
  double result;
  result = mResult(0,0);
  // std::cout << "Returning output of constraintFunc_RESCLF2" << std::endl;
  return result;

}

void constraintFunc_RESCLF_Torque(unsigned m, double* result, unsigned n, const double* x,
                    double* grad, void* f_data) {

   //n is the length of x, m is the length of result. 
  // The n dimension of grad is stored contiguously, so that \dci/\dxj is stored in grad[i*n + j]
    //Here you see take dCi/dx0...dxn and store it one by one, then repeat. grad is just an one dimensional array

  OptParams* constParams = reinterpret_cast<OptParams*>(f_data);

  // std::cout << "done reading constraintFunc_RESCLF_Torque constParams " << std::endl;
  // std::cout << "m = " << m << "      n = " << n << std::endl; // m = 7 and n = 7 here
  // std::cout << "size of constParams->P = " << constParams->P.rows() << "*" << constParams->P.cols() << std::endl; //P = 7*6

  if (grad != NULL) {

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        // Since j = 0 imply x(0) which is relaxation term and is zero in this case
        if(j != 0) 
          grad[i * n + j] = constParams->P(i, j-1);
        else
          grad[i * n + j] = 0;
      }
    }
  }
  // std::cout << "done with gradient" << std::endl;

  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (size_t i = 0; i < n; i++) X(i) = x[i];
  // std::cout << "done reading x" << std::endl;

  Eigen::VectorXd mResult;

// Match dimension here. X.tail(6) is used since we are considering position and orientation for one arm. This can 
// change as per the requirement.
  mResult = constParams->P * X.tail(6) - constParams->b;
  for (size_t i = 0; i < m; i++) {
    result[i] = mResult(i);
  }
  // std::cout << "done calculating the result in constraintFunc_RESCLF_Torque" << std::endl;
}

double constraintFunc_RESCLF_CBF(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
  OptParams_CBF* constParams = reinterpret_cast<OptParams_CBF*>(my_func_data);
  // std::cout << "done reading optParams in constraintFunc_RESCLF_CBF" << std::endl;
  
  size_t n = x.size();
  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) X(i) = x[i];

  // std::cout << "Size of constraintFunc_RESCLF_CBF constParams->L_G.size() = " << constParams->L_G.rows() << "*" << constParams->L_G.cols() << std::endl;
  // std::cout << "x = " << n << std::endl;

  if (!grad.empty()) {
    Eigen::MatrixXd mGrad = Eigen::VectorXd::Zero(n);
  // Gradient has to be negative since barrier is positive as per the formulation    
    for(int i = 1;i < n ;i++){
      mGrad(i,0) = -constParams->L_G(0,i-1);
    }
    
    // std::cout << "constraintFunc_RESCLF_CBF Check 2" << std::endl;
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
  }

  Eigen::Matrix<double,1 ,1> mResult;
  // Match dimension here. X.tail(6) is used since we are considering position and orientation for one arm. This can 
  // change as per the requirement.
  mResult = constParams->L_G * X.tail(6) + constParams->L_F + constParams->K * constParams->eta;
  // std::cout << "constraintFunc_CBF Check 3" << std::endl;
  double result;

  //Send the negative of the result computed since the formulation is positive
  result = -mResult(0,0);
  // std::cout << "Returning result" << std::endl;
  return result;

}

// ********************************** Munzir Optimization Framework ***************************

//==============================================================================
void constraintFunc(unsigned m, double* result, unsigned n, const double* x,
                    double* grad, void* f_data) {
  OptParams* constParams = reinterpret_cast<OptParams*>(f_data);
  // std::cout << "done reading optParams " << std::endl;

  if (grad != NULL) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        grad[i * n + j] = constParams->P(i, j);
      }
    }
  }
  // std::cout << "done with gradient" << std::endl;

  Eigen::MatrixXd X = Eigen::VectorXd::Zero(n);
  for (size_t i = 0; i < n; i++) X(i) = x[i];
  // std::cout << "done reading x" << std::endl;

  Eigen::VectorXd mResult;
  mResult = constParams->P * X - constParams->b;
  for (size_t i = 0; i < m; i++) {
    result[i] = mResult(i);
  }
  // std::cout << "done calculating the result"
}

//==============================================================================
double optFunc(const std::vector<double>& x, std::vector<double>& grad,
               void* my_func_data) {
  OptParams* optParams = reinterpret_cast<OptParams*>(my_func_data);
  // std::cout << "done reading optParams " << std::endl;
  // Eigen::Matrix<double, 18, 1> X(x.data());
  size_t n = x.size();
  Eigen::VectorXd X = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) X(i) = x[i];
  // std::cout << "done reading x" << std::endl;

  if (!grad.empty()) {
    Eigen::MatrixXd mGrad =
        optParams->P.transpose() * (optParams->P * X - optParams->b);
    // std::cout << "done calculating gradient" << std::endl;
    Eigen::VectorXd::Map(&grad[0], mGrad.size()) = mGrad;
    // std::cout << "done changing gradient cast" << std::endl;
  }
  // std::cout << "about to return something" << std::endl;
  return (0.5 * pow((optParams->P * X - optParams->b).norm(), 2));
}

// ==============================Defining Function to display ===============
void displayAllFrames(
    const dart::simulation::WorldPtr& world, 
    const dart::dynamics::SkeletonPtr& robot,
    std::size_t index)
{
      //Interactive Frame contains three axes, three planes, and three circles. The code below turns plane and circles off.
    // InteractiveFramePtr frame = std::make_shared<InteractiveFrame>(dart::dynamics::Frame::World());
  std::cout << "displayAllFrames Check 1" << std::endl;
  if(index != 100)
  {

    BodyNode* bn = robot->getBodyNode(index);
    // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>(bn, bn->getName()+"/frame"));

    InteractiveFramePtr frame = std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, bn->getName()+"/frame");

    // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, bn->getName()+"/frame"));
    world->addSimpleFrame(frame);
    // Interactive Frame contains three axes, three planes, and three circles. The code below turns plane and circles off.
    for(const auto type : {InteractiveTool::ANGULAR, InteractiveTool::PLANAR})
    {
        for(std::size_t i=0; i < 3; ++i)
        {
          frame->getTool(type, i)->setEnabled(false);
        }
    }

    for(std::size_t j=0; j < bn->getNumChildJoints(); ++j)
    {
      const Joint* joint = bn->getChildJoint(j);
      const Eigen::Isometry3d offset = joint->getTransformFromParentBodyNode();
      // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>(bn, joint->getName()+"/frame", offset));
      world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, joint->getName()+"/frame", offset));
      // Interactive Frame contains three axes, three planes, and three circles. The code below turns plane and circles off.
      for(const auto type : {InteractiveTool::ANGULAR, InteractiveTool::PLANAR})
      {
        for(std::size_t i=0; i < 3; ++i)
        {
          frame->getTool(type, i)->setEnabled(false);
        }
      }
    
    }
  }
  else
  {
    for(std::size_t i=0; i < robot->getNumBodyNodes(); ++i)
    {
      BodyNode* bn = robot->getBodyNode(i);
      // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>(bn, bn->getName()+"/frame"));

      InteractiveFramePtr frame = std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, bn->getName()+"/frame");

      // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, bn->getName()+"/frame"));
      world->addSimpleFrame(frame);
      // Interactive Frame contains three axes, three planes, and three circles. The code below turns plane and circles off.
      for(const auto type : {InteractiveTool::ANGULAR, InteractiveTool::PLANAR})
      {
          for(std::size_t i=0; i < 3; ++i)
          {
            frame->getTool(type, i)->setEnabled(false);
          }
      }

      for(std::size_t j=0; j < bn->getNumChildJoints(); ++j)
      {
        const Joint* joint = bn->getChildJoint(j);
        const Eigen::Isometry3d offset = joint->getTransformFromParentBodyNode();
        // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>(bn, joint->getName()+"/frame", offset));
        // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, joint->getName()+"/frame", offset));
        InteractiveFramePtr frame = std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, joint->getName()+"/frame", offset);
        world->addSimpleFrame(frame);
        // Interactive Frame contains three axes, three planes, and three circles. The code below turns plane and circles off.
        for(const auto type : {InteractiveTool::ANGULAR, InteractiveTool::PLANAR})
        {
          for(std::size_t i=0; i < 3; ++i)
          {
            frame->getTool(type, i)->setEnabled(false);
          }
        }
      
      }
    } 
  }
  
}


// ==============================Defining reference frame to display ===============

// Not working at the moment. Will need to spend some time fixing the issues.
// void displayRefFrames(
//     const dart::simulation::WorldPtr& world, 
//     const SimpleFramePtr& RPY_Frame)
// {
      
//   //Commenting it to set of the interactive frame later.
//   // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>(bn, bn->getName()+"/frame")); 
//   InteractiveFramePtr frame = std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)RPY_Frame, bn->getName()+"/frame");

//   // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, bn->getName()+"/frame"));
//   world->addSimpleFrame(frame);

//   for(const auto type : {InteractiveTool::ANGULAR, InteractiveTool::PLANAR})
//   {
//       for(std::size_t i=0; i < 3; ++i)
//       {
//         frame->getTool(type, i)->setEnabled(false);
//       }
//   }

//   for(std::size_t j=0; j < bn->getNumChildJoints(); ++j)
//   {
//     const Joint* joint = bn->getChildJoint(j);
//     const Eigen::Isometry3d offset = joint->getTransformFromParentBodyNode();
//     // world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>(bn, joint->getName()+"/frame", offset));
//     world->addSimpleFrame(std::make_shared<dart::gui::osg::InteractiveFrame>((dart::dynamics::Frame*)bn, joint->getName()+"/frame", offset));
//     for(const auto type : {InteractiveTool::ANGULAR, InteractiveTool::PLANAR})
//     {
//       for(std::size_t i=0; i < 3; ++i)
//       {
//         frame->getTool(type, i)->setEnabled(false);
//       }
//     }
  
//   }
  
// }

class OperationalSpaceControlWorld : public dart::gui::osg::WorldNode
{
public:
  OperationalSpaceControlWorld(dart::simulation::WorldPtr _world)
    : dart::gui::osg::WorldNode(_world)
  {
    // Extract the relevant pointers
    mRobot = mWorld->getSkeleton(0);
    mEndEffector = mRobot->getBodyNode(mRobot->getNumBodyNodes() - 1);
    
    // Setup gain matrices
    std::size_t dofs = mEndEffector->getNumDependentGenCoords();
    // std::size_t dofs =  mRobot->getNumJoints();

    
    mKp.setZero();
    for (std::size_t i = 0; i < 3; ++i){
      mKp(i, i) = 1500.0;
      mKv(i, i) = 700.0;
    }

    mKd.setZero(dofs, dofs);
    for (std::size_t i = 0; i < dofs; ++i)
      mKd(i, i) = 5.0;

    // Set joint properties
    for (std::size_t i = 0; i < mRobot->getNumJoints(); ++i)
    {
      // mRobot->getJoint(i)->setLimitEnforcement(false);
      mRobot->getJoint(i)->setDampingCoefficient(0, 0.5);
    }

    mOffset = Eigen::Vector3d(0.001, 0, 0);
    

    // Create target Frame
    Eigen::Isometry3d tf = mEndEffector->getWorldTransform();
    tf.pretranslate(mOffset);
    mTarget = std::make_shared<SimpleFrame>(Frame::World(), "target", tf);
    ShapePtr ball(new SphereShape(0.025));
    mTarget->setShape(ball);
    mTarget->getVisualAspect(true)->setColor(Eigen::Vector3d(0.9, 0, 0));
    mWorld->addSimpleFrame(mTarget);
    cout << "Constructor: mTarget Position = \n" << mTarget->getWorldTransform().translation() << endl;

//  *************************** Adding Obstacle in Similar to mTarget Format ************
    mObscaleOffset = Eigen::Vector3d(0.15, 0.01, 0);
    mObscaleRadius = 0.035; //0.11;

    Eigen::Isometry3d tf1 = mEndEffector->getWorldTransform();
    tf1.prerotate(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()))
      .prerotate(Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()))
      .prerotate(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
    tf1.pretranslate(mObscaleOffset);
    mObscale = std::make_shared<SimpleFrame>(Frame::World(), "obstacle", tf1);
    ShapePtr ObstacleBall(new SphereShape(mObscaleRadius));
    mObscale->setShape(ObstacleBall);
    mObscale->getVisualAspect(true)->setColor(Eigen::Vector3d(0.9, 0.9, 0));
    mWorld->addSimpleFrame(mObscale);
    mObscaleInitPos = mObscale->getWorldTransform().translation();
    cout << "Constructor: mObscale Position = \n" << mObscale->getWorldTransform().translation() << endl;

//  ********************************************************************************

    mOffset
        = mEndEffector->getWorldTransform().rotation().transpose() * mOffset;

// ******************************************** Loading F, g, and P matrix *************

    mP = readMatrix("/home/krang/SethResearch/operational_space_control_SepareteRESCLF_IIWA/P.txt");
    mF = readMatrix("/home/krang/SethResearch/operational_space_control_SepareteRESCLF_IIWA/F.txt");
    mG = readMatrix("/home/krang/SethResearch/operational_space_control_SepareteRESCLF_IIWA/G.txt"); 
    
// Change the dimension here for the number of decision variable. Currently its 7, 3 for position, 3 for orientation 
// and 1 for relaxation term.
    mddqBodyRef = Eigen::VectorXd::Zero(7); 
    cout << "Done Reading mP, MF and MG matrix" << endl;

// ******************************************Declaring variables for dumping values **********
    ee_position.open("ee_position.csv");
    joint_position.open("joint_position.csv");
    torque_value.open("torque_value.csv");
    target_position.open("target_position.csv");
    barrier_position.open("barrier_position.csv");
    cout << "Done Initializing Files" << endl;

// ******************************** Orientation Related initialization and representation ********
    mKpOr.setZero();
    mKvOr.setZero();
    for (std::size_t i = 0; i < 3; ++i){
      mKpOr(i, i) = 750.0;
      mKvOr(i, i) = 250.0;
    }

    //Define Orientation of mEndEffector and then ensure it donot change. Forget about changing the reference for now
    mTargetRPY.setZero();
    mTargetRPY = dart::math::matrixToEulerXYZ(mEndEffector->getTransform().rotation()); 
    displayAllFrames(_world, mRobot, mRobot->getNumBodyNodes() - 1);
    // displayAllFrames(_world, mRobot, 100); // 100 to indicate that you should display all frames
  }

  // Triggered at the beginning of each simulation step
  void customPreStep() override
  {
    Eigen::MatrixXd M = mRobot->getMassMatrix();

    LinearJacobian Jv = mEndEffector->getLinearJacobian(mOffset);
    Eigen::MatrixXd pinv_Jv
        = Jv.transpose()
          * (Jv * Jv.transpose() + 0.0025 * Eigen::Matrix3d::Identity())
                .inverse();

    LinearJacobian dJv = mEndEffector->getLinearJacobianDeriv(); //mEndEffector->getLinearJacobianDeriv(mOffset);
    Eigen::MatrixXd pinv_dJv
        = dJv.transpose()
          * (dJv * dJv.transpose() + 0.0025 * Eigen::Matrix3d::Identity())
                .inverse();

    Eigen::Vector3d e = mTarget->getWorldTransform().translation()
                        - mEndEffector->getWorldTransform() * mOffset;

    // Eigen::Vector3d e = mTarget->getWorldTransform().translation()
    //                     - mEndEffector->getWorldTransform().translation();

    Eigen::Vector3d de = -mEndEffector->getLinearVelocity(mOffset);

    Eigen::VectorXd Cg = mRobot->getCoriolisAndGravityForces();

    Eigen::VectorXd T_OriginalController = M * (pinv_Jv * mKp * de + pinv_dJv * mKp * e) + Cg
              + mKd * pinv_Jv * mKp * e;


    // mForces = M * (pinv_J * mKp * de + pinv_dJ * mKp * e) + Cg
    //           + mKd * pinv_J * mKp * e;

// *************************Orientation Control Part ***********************

      // End-effector Orientation
    Eigen::Quaterniond quat(mEndEffector->getTransform().rotation());
    double quat_w = quat.w(); 
    Eigen::Vector3d quat_xyz(quat.x(), quat.y(), quat.z());
    if(quat_w < 0) {quat_w *= -1.0; quat_xyz *= -1.0; }
    Eigen::Quaterniond quatRef(Eigen::AngleAxisd(mTargetRPY(0), Eigen::Vector3d::UnitX()) *
             Eigen::AngleAxisd(mTargetRPY(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(mTargetRPY(2), Eigen::Vector3d::UnitZ()));
    double quatRef_w = quatRef.w(); 
    Eigen::Vector3d quatRef_xyz(quatRef.x(), quatRef.y(), quatRef.z());
    if(quatRef_w < 0) { quatRef_w *= -1.0; quatRef_xyz *= -1.0; }
    Eigen::Vector3d quatError_xyz = quatRef_w*quat_xyz - quat_w*quatRef_xyz + quatRef_xyz.cross(quat_xyz);
    // double quatError_w = quat_w*quatRef_w - quat_xyz.dot(quatRef_xyz);
    Eigen::Vector3d w = mEndEffector->getAngularVelocity();
    // Eigen::Vector3d dwref = -mKpOr*quatError_xyz/(2*quatError_w) - mKvOr*w;
    Eigen::Vector3d dwref = -mKpOr*quatError_xyz - mKvOr*w;
    AngularJacobian Jw = mEndEffector->getAngularJacobian();       // 3 x n
    AngularJacobian dJw = mEndEffector->getAngularJacobianDeriv();  // 3 x n

    Eigen::MatrixXd pinv_Jw
        = Jw.transpose()
          * (Jw * Jw.transpose() + 0.0025 * Eigen::Matrix3d::Identity())
                .inverse();


    // Eigen::Matrix<double, 3, 7> POr = Jw;
    // Eigen::Vector3d bOr = -(dJw*dq - dwref);


// ***************** Writing our own controller *******************************
    // // Introducing some changes to go with maths
    Eigen::VectorXd dq  = mRobot->getVelocities();                 // 6 x 1
    Eigen::VectorXd x   = mEndEffector->getWorldTransform() * mOffset;  // 3 * 1
    Eigen::Vector3d dx  = mEndEffector->getLinearVelocity(mOffset);  // 3 * 1
    // cout << "Size of dq = " << dq.rows() << "*" << dq.cols() << endl; // Size is 6 x 1
    Eigen::MatrixXd M2_position = M*pinv_Jv;
    // cout << "Size of M2 = " << M2.rows() << "*" << M2.cols() << endl;
    // cout << "Size of e = " << e.rows() << "*" << e.cols() << endl;
    // cout << "mTarget Position = \n" << mTarget->getWorldTransform().translation() << endl;
    // cout << "mEndEffector Position = \n" << mEndEffector->getWorldTransform() * mOffset << endl;

    Eigen::Vector3d desired_ddx_pos = mKp*e + mKv*de;
    Eigen::VectorXd T_OurFormulation = M2_position*(desired_ddx_pos - dJv*dq) + Cg;



    

// ********************** Writing RESCLF Controller with optimization *****************
  // Eigen::VectorXd dq  = mRobot->getVelocities();     // 6 x 1
  // Eigen::Vector3d x = mEndEffector->getWorldTransform().translation();
  // Eigen::Vector3d dx = mEndEffector->getLinearVelocity();

  Eigen::Vector3d x_t = mTarget->getWorldTransform().translation();

  Eigen::Vector3d x_ee = mEndEffector->getWorldTransform().translation();

  Eigen::Vector3d e_t =  x_t - mEndEffector->getWorldTransform().translation();

  Eigen::Vector3d de_t = mTarget->getLinearVelocity() - mEndEffector->getLinearVelocity();

  // std::cout << "Dimension of joint = " << dq.rows() << "*" << dq.cols() << std::endl;

  size_t n = mF.rows();
  // std::cout << "n = mF.rows() = " << n <<  std::endl;

  Eigen::Matrix<double, 6,1> EE_Eta, Or_Eta;
  EE_Eta << e_t, de_t; //x - _targetPosition,dx;
  Or_Eta << quatError_xyz, w;
  std::cout << "Check 1" <<  std::endl;

// Defining eta for error
  Eigen::VectorXd mEta1(n), mEta2(n);
  mEta1 << EE_Eta;
  mEta2 << Or_Eta;

// Defining Augmented Jacobian
  Eigen::Matrix<double, 6,7> J_Aug, dJ_Aug;
  J_Aug << Jv, Jw;
  dJ_Aug << dJv, dJw;

//Defining Augmented Jacobian pseudo inverse
  Eigen::MatrixXd pinv_J_Aug
        = J_Aug.transpose()
          * (J_Aug * J_Aug.transpose() + 0.0025 * Eigen::MatrixXd::Identity(6,6))
                .inverse();

// Defining desired ddx_augment
  Eigen::Matrix<double, 6, 1> desired_ddx_Aug;
  desired_ddx_Aug << desired_ddx_pos, dwref;


  std::cout << "Check 2" <<  std::endl;

// Defining the LfV_x, LgV_x and V_x for position
  Eigen::MatrixXd LfV_x_pos = mEta1.transpose()*(mF.transpose()*mP+mP*mF)*mEta1;
  Eigen::MatrixXd LgV_x_pos = 2*mEta1.transpose()*mP*mG;
  Eigen::MatrixXd V_x_pos = mEta1.transpose()*mP*mEta1;

// Defining the LfV_x, LgV_x and V_x for orientation
  Eigen::MatrixXd LfV_x_ori = mEta2.transpose()*(mF.transpose()*mP+mP*mF)*mEta2;
  Eigen::MatrixXd LgV_x_ori = 2*mEta2.transpose()*mP*mG;
  Eigen::MatrixXd V_x_ori = mEta2.transpose()*mP*mEta2;




  Eigen::Matrix<double, 7,1> mTauLim;
  mTauLim << 320, 320, 176, 176, 110, 40, 40; // Need to be adjusted when adding limits

  OptParams_RESCLF optParams_RESCLF;
 
  double lambda_minQ = 1;  // Provided by the Matlab QQ Matrix
  double lambda_maxP = 2.7321; /// Provided by the Matlab P Matrix

  OptParams_RESCLF inequalityconstraintParams_RESCLF;
  //Declaring variable for position. done in the same data structure because of NLOPT objective function requirement. 
  inequalityconstraintParams_RESCLF.L_F_pos = LfV_x_pos;
  inequalityconstraintParams_RESCLF.L_G_pos = LgV_x_pos;
  inequalityconstraintParams_RESCLF.V_x_pos = V_x_pos;

  //Declaring variable for orienation
  inequalityconstraintParams_RESCLF.L_F_ori = LfV_x_ori;
  inequalityconstraintParams_RESCLF.L_G_ori = LgV_x_ori;
  inequalityconstraintParams_RESCLF.V_x_ori = V_x_ori;

  //Declaring variable for relaxation
  inequalityconstraintParams_RESCLF.gamma = 0.1;//lambda_minQ/lambda_maxP;
  inequalityconstraintParams_RESCLF.relaxation = 1e6;

  //Adding Barrier Stuff
  // Obtaining the obstacle information and EndEffector Information
  Eigen::Vector3d x_b = mObscale->getWorldTransform().translation();
  // Eigen::Vector3d x = mEndEffector->getWorldTransform().translation();
  // Eigen::Vector3d dx = mEndEffector->getLinearVelocity();

  // Eigen::Vector3d x_t = mTarget->getWorldTransform().translation();

  Eigen::Vector3d e_b = mObscale->getWorldTransform().translation()
                      - mEndEffector->getWorldTransform().translation();

  Eigen::Vector3d de_b = mObscale->getLinearVelocity()-mEndEffector->getLinearVelocity();

  Eigen::Vector3d ddx_b = mObscale->getLinearAcceleration();

  Eigen::MatrixXd invM = M.colPivHouseholderQr().inverse();

  // Define an A Matrix for ellipsoid 
  Eigen::Matrix<double, 3,3> A = Eigen::MatrixXd::Zero(3,3);
  double mObscaleRadius1 = 0.1;
  A(0,0) = 1/(mObscaleRadius1*mObscaleRadius1);
  A(1,1) = 1/(mObscaleRadius1*mObscaleRadius1);
  A(2,2) = 1/(mObscaleRadius1*mObscaleRadius1);

  // Degine Eta for Exponential Barrier Function
  Eigen::Matrix<double,2,1> eta;
  eta << e_b.transpose()*A*e_b - 1,2*e_b.transpose()*A*de_b;

  // Define K matrix containing poles
  double p0, p1, h_0, h_1, temp1;
  h_0 = e_b.transpose()*A*e_b - 1;
  h_1 = 2*e_b.transpose()*A*de_b;
  temp1 = h_1/h_0;
  p1 = 50 + temp1;
  p0 = 500;

  Eigen::Matrix<double,1,2> K_b;
  // K_b << 500,80;
  K_b << 2000,800;
// std::cout << "Check 3" <<  std::endl;
  OptParams_CBF CBF;

  CBF.L_G = -2*(e_b).transpose()*A*Jv*invM*M*pinv_J_Aug; //2*(x - x_b).transpose()*A*J*M.colPivHouseholderQr().inverse();
  CBF.L_F = 2*e_b.transpose()*A*ddx_b + 2*e_b.transpose()*A*de_b - 2*e_b.transpose()*A*(dJv*dq - Jv*invM*Cg);
  -2*e_b.transpose()*A*Jv*invM*(M*pinv_J_Aug*(desired_ddx_Aug - dJ_Aug*dq)+Cg); //2*de_b.transpose()*A*de_b + 2*e_b.transpose()*A*(dJv*dq - Jv*invM*Cg);
  CBF.eta = eta;
  CBF.K = K_b;
  // CBF.u_nom = bodyTorques1;


  // std::cout << "Check 3_1" <<  std::endl;

  //Declaring constraint for Torque constraints
  OptParams inequalityconstraintParams1[2];

  const std::vector<double> inequalityconstraintTol1(7, 1e-5);
  
 // Implement Torue limit as inequality constraint. Function is implemented as of form  min_x Px-b
  inequalityconstraintParams1[0].P = M*pinv_J_Aug;
  inequalityconstraintParams1[1].P = -M*pinv_J_Aug;
  inequalityconstraintParams1[0].b = -(M*pinv_J_Aug*(desired_ddx_Aug - dJ_Aug*dq) + Cg - mTauLim); //-sign due to - with b.
  inequalityconstraintParams1[1].b = -(-M*pinv_J_Aug*(desired_ddx_Aug - dJ_Aug*dq) - Cg - mTauLim);

  // nlopt::opt opt1(nlopt::LN_COBYLA, mOptDim);
  nlopt::opt opt1(nlopt::LD_SLSQP, 7);
  // nlopt::opt opt1(nlopt::AUGLAG, 7);

  double minf1;
  opt1.set_min_objective(optFunc_RESCLF, &inequalityconstraintParams_RESCLF);
  opt1.add_inequality_constraint(constraintFunc_RESCLF1, &inequalityconstraintParams_RESCLF,1e-6);
  opt1.add_inequality_constraint(constraintFunc_RESCLF2, &inequalityconstraintParams_RESCLF,1e-6);

  opt1.add_inequality_mconstraint(constraintFunc_RESCLF_Torque, &inequalityconstraintParams1[0],
                                 inequalityconstraintTol1);
  opt1.add_inequality_mconstraint(constraintFunc_RESCLF_Torque, &inequalityconstraintParams1[1],
                                 inequalityconstraintTol1);
  opt1.add_inequality_constraint(constraintFunc_RESCLF_CBF, &CBF, 1e-3);

  std::vector<double> ddqBodyRef_vec1(7);
  Eigen::VectorXd::Map(&ddqBodyRef_vec1[0], mddqBodyRef.size()) = mddqBodyRef;
  // std::cout << "size of mddqBodyRef = " << mddqBodyRef.rows() << "*" << mddqBodyRef.cols() <<std::endl;
  try {
    std::cout << "Check 4" <<  std::endl;
    opt1.set_xtol_rel(1e-4);
    opt1.set_maxtime(0.01);
    nlopt::result result = opt1.optimize(ddqBodyRef_vec1, minf1);
    std::cout << "Check 5" <<  std::endl;
  } 
  catch (std::exception& e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
  }

  Eigen::Matrix<double, 6, 1> ddq_RESCLF;
  for(int i = 1;i < 7;i++) ddq_RESCLF(i-1,0) = ddqBodyRef_vec1[i]; 

  // std::cout << "Check 6" <<  std::endl;
  Eigen::VectorXd bodyTorques1;
  
  bodyTorques1 = M*pinv_J_Aug*(desired_ddx_Aug + ddq_RESCLF - dJ_Aug*dq) + Cg; //A_LgLfy.colPivHouseholderQr().solve(-Lf_yx + ddq1);
  
   
// *******************************Munzir Controller ************************************
 /* double wPos = 1, wOr = 1;

  Eigen::Matrix<double, 3, 7> PPos = Jv;
  Eigen::Vector3d bPos = -(dJv*dq - desired_ddx_pos);

  Eigen::Matrix<double, 3, 7> POr = Jw;
  Eigen::Vector3d bOr = -(dJw*dq - dwref);


   // Optimizer stuff
  nlopt::opt opt(nlopt::LD_MMA, 7);
  OptParams optParams;
  std::vector<double> ddq_vec(7);
  double minf;

  // Perform optimization to find joint accelerations 
  Eigen::MatrixXd P(PPos.rows() + POr.rows(), PPos.cols() );
  P << wPos*PPos,
       wOr*POr;
  
  Eigen::VectorXd b(bPos.rows() + bOr.rows(), bPos.cols() );
  b << wPos*bPos,
       wOr*bOr;
       
  optParams.P = P;
  optParams.b = b;
  opt.set_min_objective(optFunc, &optParams);
  opt.set_xtol_rel(1e-4);
  opt.set_maxtime(0.005);
  opt.optimize(ddq_vec, minf);
  Eigen::Matrix<double, 7, 1> ddq(ddq_vec.data()); 
  

  //torques
  mForces = M*ddq + Cg;

  std::cout << "bodyTorques from RESCLF = \n" << bodyTorques1 << std::endl;
  std::cout << "bodyTorques from Munzir = \n" << mForces << std::endl;

  // Apply the joint space forces to the robot
  // mRobot->setForces(T_OriginalController);
  // mRobot->setForces(bodyTorques1);
  // mRobot->setForces(mForces);


// **************************************************************************    

// ******************* Control Barrier Function *****************************

    // Obtaining the obstacle information and EndEffector Information
    Eigen::Vector3d x_b = mObscale->getWorldTransform().translation();
    // Eigen::Vector3d x = mEndEffector->getWorldTransform().translation();
    // Eigen::Vector3d dx = mEndEffector->getLinearVelocity();

    // Eigen::Vector3d x_t = mTarget->getWorldTransform().translation();

    Eigen::Vector3d e_b = mObscale->getWorldTransform().translation()
                        - mEndEffector->getWorldTransform().translation();

    Eigen::Vector3d de_b = mObscale->getLinearVelocity()-mEndEffector->getLinearVelocity();

    Eigen::Vector3d ddx_b = mObscale->getLinearAcceleration();

    Eigen::MatrixXd invM = M.colPivHouseholderQr().inverse();

    // Define an A Matrix for ellipsoid 
    Eigen::Matrix<double, 3,3> A = Eigen::MatrixXd::Zero(3,3);
    double mObscaleRadius1 = 0.1;
    A(0,0) = 1/(mObscaleRadius1*mObscaleRadius1);
    A(1,1) = 1/(mObscaleRadius1*mObscaleRadius1);
    A(2,2) = 1/(mObscaleRadius1*mObscaleRadius1);

    // Degine Eta for Exponential Barrier Function
    Eigen::Matrix<double,2,1> eta;
    eta << e_b.transpose()*A*e_b - 1,2*e_b.transpose()*A*de_b;

    // Define K matrix containing poles
    double p0, p1, h_0, h_1, temp1;
    h_0 = e_b.transpose()*A*e_b - 1;
    h_1 = 2*e_b.transpose()*A*de_b;
    temp1 = h_1/h_0;
    p1 = 50 + temp1;
    p0 = 500;

    Eigen::Matrix<double,1,2> K_b;
    // K_b << 500,80;
    K_b << 2000,800;

    OptParams_CBF CBF;

    CBF.L_G = -2*(e_b).transpose()*A*Jv*invM; //2*(x - x_b).transpose()*A*J*M.colPivHouseholderQr().inverse();
    CBF.L_F = 2*e_b.transpose()*A*ddx_b + 2*e_b.transpose()*A*de_b - 2*e_b.transpose()*A*(dJv*dq - Jv*invM*Cg); //2*de_b.transpose()*A*de_b + 2*e_b.transpose()*A*(dJv*dq - Jv*invM*Cg);
    CBF.eta = eta;
    CBF.K = K_b;
    CBF.u_nom = bodyTorques1;

    // nlopt::opt opt(nlopt::LD_MMA, 7);
    nlopt::opt opt_CBF(nlopt::LD_SLSQP, 7);
    // nlopt::opt opt_CBF(nlopt::AUGLAG, 7);

    std::vector<double> ddq_vec_CBF(7, 0.0);
    std::vector<double> CBF_TorqueLimit(7, 0.0);
    //Map Torque limit to std vector
    Eigen::VectorXd::Map(&CBF_TorqueLimit[0], mTauLim.size()) = -mTauLim; 
    // for(int i = 0; i < 7; i++)
    //   std::cout << "CBF_TorqueLimit[ " << i << "] = " << CBF_TorqueLimit[i] << std::endl;

    // Eigen::VectorXd::Map(&ddq_vec_CBF[0], T_OriginalController.size()) = T_OriginalController;
    double minf_CBF;

    opt_CBF.set_min_objective(optFunc_CBF, &CBF);
    opt_CBF.add_inequality_constraint(constraintFunc_CBF, &CBF, 1e-3);
    opt_CBF.set_lower_bounds(CBF_TorqueLimit);
    Eigen::VectorXd::Map(&CBF_TorqueLimit[0], mTauLim.size()) = mTauLim;
    opt_CBF.set_upper_bounds(CBF_TorqueLimit); 
    opt_CBF.set_xtol_rel(1e-4);

    try{
      nlopt::result result = opt_CBF.optimize(ddq_vec_CBF, minf_CBF);
    }
    catch(std::exception &e){
      std::cout << "nlopt failed: " << e.what() << std::endl;
    }

    Eigen::Matrix<double, 7, 1> ddq1(ddq_vec_CBF.data()); 
    
    std::cout << "x_pos = \n" << x << std::endl;
    std::cout << "x_obstacle = \n" << x_b << std::endl;
    std::cout << "e_b = \n" << e_b << std::endl;
    // std::cout << "GlobalInit obstacle = \n" << mObscaleInitPos << std::endl;
 
    // std::cout << "x_pos(1) = " << setprecision(4) << x(0) << "            x_t(0) = " << setprecision(4) << x_t(0) << "          x_b(0) = " << setprecision(4) << x_b(0) <<std::endl;
    // std::cout << "           " << setprecision(4) << x(1) << "                     " << setprecision(4) << x_t(1) << "                   " << setprecision(4) << x_b(1) <<std::endl;
    // std::cout << "           " << setprecision(4) << x(2) << "                     " << setprecision(4) << x_t(2) << "                   " << setprecision(4) << x_b(2) <<std::endl;
  
    std::cout << "Objective Value = " << minf_CBF << std::endl;
    std::cout << "Barrier Value e_b.transpose()*A*e_b - 1 = " << e_b.transpose()*A*e_b - 1 << std::endl;
    std::cout << "Barrier Value 2*e_b.transpose()*A*de_b = " << 2*e_b.transpose()*A*de_b << std::endl;

    // std::cout << "Barrier Value 1-e_b.transpose()*A*e_b = " << 1 - e_b.transpose()*A*e_b << std::endl;
    // std::cout << "Barrier Value -2*e_b.transpose()*A*de_b = " << 2*e_b.transpose()*A*de_b << std::endl;

    std::cout << "bodyTorques from CBF = \n" << ddq1 << std::endl;
    std::cout << "bodyTorques from Default = \n" << T_OriginalController << std::endl;


*/
// **************************************************************************

    ee_position << x_ee(0) << ", " << x_ee(1) << ", " << x_ee(2) << endl;
    target_position << x_t(0) << ", " << x_t(1) << ", " << x_t(2) << endl;
    barrier_position << x_b(0) << ", " << x_b(1) << ", " << x_b(2) << endl;
    torque_value << bodyTorques1(0) << ", " << bodyTorques1(1) << ", " << bodyTorques1(2) << ", "
                << bodyTorques1(3) << ", " << bodyTorques1(4) << ", " << bodyTorques1(5) << ", "
                << bodyTorques1(6) << endl;


    mForces = bodyTorques1;//T_OurFormulation;
    mRobot->setForces(mForces);
  
  }

  dart::gui::osg::DragAndDrop* dnd;
  dart::gui::osg::DragAndDrop* dndObstacle;

  ~OperationalSpaceControlWorld() {
    ee_position.close();
    joint_position.close();
    torque_value.close();
    target_position.close();
    barrier_position.close();   
 
  }
protected:
  // Triggered when this node gets added to the Viewer
  void setupViewer() override
  {
    if (mViewer)
    {
      dnd = mViewer->enableDragAndDrop(mTarget.get());
      dnd->setObstructable(false);
      mViewer->addInstructionText(
          "\nClick and drag the red ball to move the target of the operational "
          "space controller\n");
      mViewer->addInstructionText(
          "Hold key 1 to constrain movements to the x-axis\n");
      mViewer->addInstructionText(
          "Hold key 2 to constrain movements to the y-axis\n");
      mViewer->addInstructionText(
          "Hold key 3 to constrain movements to the z-axis\n");


    // **************************** Adding Obstacle Setting *******************
      dndObstacle = mViewer->enableDragAndDrop(mObscale.get());
      dndObstacle->setObstructable(false);
    }
  }

  SkeletonPtr mRobot;
  BodyNode* mEndEffector;
  SimpleFramePtr mTarget;
 
  Eigen::Vector3d mOffset;
  Eigen::Matrix3d mKp;
  Eigen::MatrixXd mKd;
  Eigen::Matrix3d mKv;
  Eigen::VectorXd mForces;

// Declaring the Obstacle 
  SimpleFramePtr mObscale;
  Eigen::Vector3d mObscaleOffset;
  Eigen::Vector3d mObscaleInitPos;
  double mObscaleRadius;

// Declaring variables for RESCLF
  Eigen::MatrixXd mF;
  Eigen::MatrixXd mG;
  Eigen::MatrixXd mP;
  Eigen::VectorXd mddqBodyRef;

// Declaring output stream variable to store variables in file
  ofstream ee_position;
  ofstream joint_position;
  ofstream torque_value;
  ofstream target_position;
  ofstream barrier_position;

// Declaring variable for orientation
  Eigen::Vector3d mTargetRPY;
  Eigen::Matrix3d mKpOr;
  Eigen::Matrix3d mKvOr;
  SimpleFramePtr mTargetRPY_Frame;

};


class ConstraintEventHandler : public ::osgGA::GUIEventHandler
{
public:
  ConstraintEventHandler(dart::gui::osg::DragAndDrop* dnd = nullptr) : mDnD(dnd)
  {
    clearConstraints();
    if (mDnD)
      mDnD->unconstrain();
  }

  void clearConstraints()
  {
    for (std::size_t i = 0; i < 3; ++i)
      mConstrained[i] = false;
  }

  virtual bool handle(
      const ::osgGA::GUIEventAdapter& ea, ::osgGA::GUIActionAdapter&) override
  {
    if (nullptr == mDnD)
    {
      clearConstraints();
      return false;
    }

    bool handled = false;
    switch (ea.getEventType())
    {
      case ::osgGA::GUIEventAdapter::KEYDOWN:
      {
        switch (ea.getKey())
        {
          case '1':
            mConstrained[0] = true;
            handled = true;
            break;
          case '2':
            mConstrained[1] = true;
            handled = true;
            break;
          case '3':
            mConstrained[2] = true;
            handled = true;
            break;
        }
        break;
      }

      case ::osgGA::GUIEventAdapter::KEYUP:
      {
        switch (ea.getKey())
        {
          case '1':
            mConstrained[0] = false;
            handled = true;
            break;
          case '2':
            mConstrained[1] = false;
            handled = true;
            break;
          case '3':
            mConstrained[2] = false;
            handled = true;
            break;
        }
        break;
      }

      default:
        return false;
    }

    if (!handled)
      return handled;

    std::size_t constraintDofs = 0;
    for (std::size_t i = 0; i < 3; ++i)
      if (mConstrained[i])
        ++constraintDofs;

    if (constraintDofs == 0 || constraintDofs == 3)
    {
      mDnD->unconstrain();
    }
    else if (constraintDofs == 1)
    {
      Eigen::Vector3d v(Eigen::Vector3d::Zero());
      for (std::size_t i = 0; i < 3; ++i)
        if (mConstrained[i])
          v[i] = 1.0;

      mDnD->constrainToLine(v);
    }
    else if (constraintDofs == 2)
    {
      Eigen::Vector3d v(Eigen::Vector3d::Zero());
      for (std::size_t i = 0; i < 3; ++i)
        if (!mConstrained[i])
          v[i] = 1.0;

      mDnD->constrainToPlane(v);
    }

    return handled;
  }

  bool mConstrained[3];

  dart::sub_ptr<dart::gui::osg::DragAndDrop> mDnD;
};

class ShadowEventHandler : public osgGA::GUIEventHandler
{
public:
  ShadowEventHandler(
      OperationalSpaceControlWorld* node, dart::gui::osg::Viewer* viewer)
    : mNode(node), mViewer(viewer)
  {
  }

  bool handle(
      const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter&) override
  {
    if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
    {
      if (ea.getKey() == 's' || ea.getKey() == 'S')
      {
        if (mNode->isShadowed())
          mNode->setShadowTechnique(nullptr);
        else
          mNode->setShadowTechnique(
              dart::gui::osg::WorldNode::createDefaultShadowTechnique(mViewer));
        return true;
      }
    }

    // The return value should be 'true' if the input has been fully handled
    // and should not be visible to any remaining event handlers. It should be
    // false if the input has not been fully handled and should be viewed by
    // any remaining event handlers.
    return false;
  }

protected:
  OperationalSpaceControlWorld* mNode;
  dart::gui::osg::Viewer* mViewer;
};

int main()
{
  dart::simulation::WorldPtr world(new dart::simulation::World);
  dart::utils::DartLoader loader;

  // Load the robot
  dart::dynamics::SkeletonPtr robot
      = loader.parseSkeleton("/home/krang/09-URDF/KUKA_IIWA/urdf/iiwa14_no_collision.urdf");
 
  world->addSkeleton(robot);
   
  // Rotate the robot so that z is upwards (default transform is not Identity)
  robot->getJoint(0)->setTransformFromParentBodyNode(Eigen::Isometry3d::Identity());

  // Load the ground
  dart::dynamics::SkeletonPtr ground
      = loader.parseSkeleton("/home/krang/Downloads/dart-6.3.1/data/urdf/KR5/ground.urdf");
  world->addSkeleton(ground);
    
  // Rotate and move the ground so that z is upwards
  Eigen::Isometry3d ground_tf
      = ground->getJoint(0)->getTransformFromParentBodyNode();
  
  ground_tf.pretranslate(Eigen::Vector3d(0, 0, 0.5));
  ground_tf.rotate(
      Eigen::AngleAxisd(constantsd::pi() / 2, Eigen::Vector3d(1, 0, 0)));
  ground->getJoint(0)->setTransformFromParentBodyNode(ground_tf);
  

  // Create an instance of our customized WorldNode
  ::osg::ref_ptr<OperationalSpaceControlWorld> node
      = new OperationalSpaceControlWorld(world);
  
  node->setNumStepsPerCycle(10);
  
  
  // Create the Viewer instance
  dart::gui::osg::Viewer viewer;
  viewer.addWorldNode(node);
  viewer.simulate(true);
  
  
  // Add our custom event handler to the Viewer
  viewer.addEventHandler(new ConstraintEventHandler(node->dnd));
  viewer.addEventHandler(new ShadowEventHandler(node.get(), &viewer));


// **************************** Adding Obstacle to this position *******************
viewer.addEventHandler(new ConstraintEventHandler(node->dndObstacle));

// ********************************************************************************


  // Print out instructions
  std::cout << viewer.getInstructions() << std::endl;

  // Set up the window to be 640x480 pixels
  viewer.setUpViewInWindow(0, 0, 720, 480);

  viewer.getCameraManipulator()->setHomePosition(
      ::osg::Vec3(2.57, 3.14, 1.64),
      ::osg::Vec3(0.00, 0.00, 0.00),
      ::osg::Vec3(-0.24, -0.25, 0.94));
  // We need to re-dirty the CameraManipulator by passing it into the viewer
  // again, so that the viewer knows to update its HomePosition setting
  viewer.setCameraManipulator(viewer.getCameraManipulator());

  // Begin the application loop
  viewer.run();
}

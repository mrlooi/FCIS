#include <tf/transform_listener.h>

inline geometry_msgs::Pose pose_axis_offset(const geometry_msgs::Pose& pose, const std::string axis, const double dist)
{
  geometry_msgs::Pose offset_pose = pose;

  tf::Quaternion q(
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z,
    pose.orientation.w);
  tf::Matrix3x3 m;
  m.setRotation(q);

  tf::Vector3 transform_vector;
  if (axis == "x")
    transform_vector=m.getColumn(0);
  else if (axis == "y")
    transform_vector=m.getColumn(2);
  else if (axis == "z")
    transform_vector=m.getColumn(1);
  else
    ROS_ERROR_STREAM("Invalid axis name");

  transform_vector.normalize();
  transform_vector=transform_vector*dist;
  offset_pose.position.x += transform_vector.getX();
  offset_pose.position.y += transform_vector.getY();
  offset_pose.position.z += transform_vector.getZ();
  
  return offset_pose;
}

inline geometry_msgs::Pose pose_axis_rotation(const geometry_msgs::Pose& pose, const std::string axis, const double angle)
{
    tf::Quaternion q;
    if (axis == "x")
      q.setRPY(angle* M_PI/180,0,0);
    else if (axis == "y")
      q.setRPY(0,angle* M_PI/180,0);
    else if (axis == "z")
      q.setRPY(0,0,angle* M_PI/180);
    else
      ROS_ERROR_STREAM("Invalid axis name");

    geometry_msgs::Pose offset_pose = pose;
    tf::Quaternion q2(0,0,0,1);
    tf::quaternionMsgToTF(offset_pose.orientation, q2);
    q2 *= q;
    q2.normalize();
    tf::quaternionTFToMsg(q2,offset_pose.orientation);

    return offset_pose;
}

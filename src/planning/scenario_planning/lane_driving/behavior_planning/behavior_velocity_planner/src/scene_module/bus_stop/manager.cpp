/*
 * Copyright 2020 Tier IV, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <scene_module/bus_stop/manager.h>

namespace
{
std::vector<lanelet::TrafficSignConstPtr> getTrafficSignRegElemsOnPath(
  const autoware_planning_msgs::PathWithLaneId & path, const lanelet::LaneletMapPtr lanelet_map)
{
  std::vector<lanelet::TrafficSignConstPtr> traffic_sign_reg_elems;

  for (const auto & p : path.points) {
    const auto lane_id = p.lane_ids.at(0);
    const auto ll = lanelet_map->laneletLayer.get(lane_id);

    const auto tss = ll.regulatoryElementsAs<const lanelet::TrafficSign>();
    for (const auto & ts : tss) {
      traffic_sign_reg_elems.push_back(ts);
    }
  }

  return traffic_sign_reg_elems;
}

std::vector<lanelet::ConstLineString3d> getStopLinesOnPath(
  const autoware_planning_msgs::PathWithLaneId & path, const lanelet::LaneletMapPtr lanelet_map)
{
  std::vector<lanelet::ConstLineString3d> bus_stops;

  for (const auto & traffic_sign_reg_elem : getTrafficSignRegElemsOnPath(path, lanelet_map)) {
    // Is stop sign?
    if (traffic_sign_reg_elem->type() != "bus_stop") {
      continue;
    }
    for (const auto & stop_line : traffic_sign_reg_elem->refLines()) {
      bus_stops.push_back(stop_line);
    }
  }

  return bus_stops;
}

std::set<int64_t> getStopLineIdSetOnPath(
  const autoware_planning_msgs::PathWithLaneId & path, const lanelet::LaneletMapPtr lanelet_map)
{
  std::set<int64_t> stop_line_id_set;

  for (const auto & stop_line : getStopLinesOnPath(path, lanelet_map)) {
    stop_line_id_set.insert(stop_line.id());
  }

  return stop_line_id_set;
}

}  // namespace

BusStopModuleManager::BusStopModuleManager() : SceneModuleManagerInterface(getModuleName()) {
  ros::NodeHandle pnh("~");
  const std::string ns(getModuleName());
  auto & p = planner_param_;
  pnh.param(ns + "/stop_margin", p.stop_margin, 0.0);
  pnh.param(ns + "/stop_check_dist", p.stop_check_dist, 2.0);
}

void BusStopModuleManager::launchNewModules(const autoware_planning_msgs::PathWithLaneId & path)
{
  msgs::BusStopArray::ConstPtr bus_stops_ = planner_data_->bus_stop_reserve;
  for (const auto & stop_line : getStopLinesOnPath(path, planner_data_->lanelet_map)) 
  {
    const auto module_id = stop_line.id();
    // std::cout << "module id : " << module_id << std::endl;
    for (int i=0; i < bus_stops_->busstops.size(); i++)
    {
      // std::cout << "bus_stops_->busstops[i].BusStopId : " << bus_stops_->busstops[i].BusStopId << std::endl;
      if (!isModuleRegistered(module_id) && module_id == bus_stops_->busstops[i].BusStopId) 
      {
        registerModule(std::make_shared<BusStopModule>(module_id, stop_line, planner_param_));
      }
    }
  }
}

std::function<bool(const std::shared_ptr<SceneModuleInterface> &)>
BusStopModuleManager::getModuleExpiredFunction(const autoware_planning_msgs::PathWithLaneId & path)
{
  const auto stop_line_id_set = getStopLineIdSetOnPath(path, planner_data_->lanelet_map);

  return [stop_line_id_set](const std::shared_ptr<SceneModuleInterface> & scene_module) {
    return stop_line_id_set.count(scene_module->getModuleId()) == 0;
  };
}

class VehicleManager:
    def __init__(self, client, tm_port, traffic_config):
        self._client = client
        self._traffic_config = traffic_config

        self._tm = self._client.get_trafficmanager(tm_port)
        self._tm.set_global_distance_to_leading_vehicle(3.5)
        self._tm.set_respawn_dormant_vehicles(True)
        self._tm.set_boundaries_respawn_dormant_vehicles(40, 100)
        self._tm.set_random_device_seed(traffic_config.tm_seed)
        print("[CARLA] Traffic Manager Port:", self._tm.get_port())

    def set_synchronous_mode(self, sync=True):
        self._tm.set_synchronous_mode(sync)

    def set_auto_lane_change(self, actor, enable):
        self._tm.auto_lane_change(actor, enable)

    def set_desired_speed(self, actor, speed):
        self._tm.set_desired_speed(actor, speed)

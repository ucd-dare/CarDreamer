def read_eval_log(file_name):
    file=open(file_name,"r")
    lines=file.readlines()

    collisions = count = avg_distance = avg_speed = wpt_distance = 0

    for line in lines:
        if line.strip()=="[CARLA] Environment reset":
            count += 1
        if "stats/sum_is_collision 1" in line:
            collisions += 1
        if "stats/mean_travel_distance" in line:
            index = line.find("stats/mean_travel_distance") + len("stats/mean_travel_distance")
            avg_distance = float(''.join(filter(str.isdigit, line[index : index + 5])))/100
        if "stats/mean_speed_norm" in line:
            index = line.find("stats/mean_speed_norm") + len("stats/mean_speed_norm")
            avg_speed = float(''.join(filter(str.isdigit, line[index : index + 5])))/100
        if "stats/mean_wpt_dis" in line:
            index = line.find("stats/mean_wpt_dis") + len("stats/mean_wpt_dis")
            wpt_distance = float(''.join(filter(str.isdigit, line[index : index + 5])))/100

    return collisions/count, avg_distance, avg_speed, wpt_distance
    

#call function
collision_rate, avg_distance, avg_speed, wpt_distance = read_eval_log('eval_log_2000.log')
print("Success Rate: ",1-collision_rate)
print("Avg. Distance: ", avg_distance)
print("Collision Rate:", collision_rate)
print("Avg. Speed: ", avg_speed)
print("Wpt. Distance: ", wpt_distance)
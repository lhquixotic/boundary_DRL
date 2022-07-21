from lib2to3.pytree import convert
from turtle import left, right
import numpy as np
import random
import math
import carla


def ransac_detection(data,distance_threshold=0.3, P=0.99, sample_size=3,
                       max_iterations=10000,lidar_height=-2.1+0.4, lidar_height_down=-2.1-0.2,
                       first_line_num=2000,alpha_threshold=0.03, use_all_sample=True,y_limit=4):
    random.seed(12345)
    max_point_num = -999
    best_model = None
    best_filt = None
    alpha = 999
    i = 0
    K = max_iterations # 增加夹角判断后，K初始值不能太小，否则可能一直没有进入最优判断语句
    # print('Start processing lidar point...')
    if not use_all_sample:
        # 不采用第一根线做法，单纯只用高度过滤
        z_filter = data[:,2] < lidar_height  # kitti 高度加0.4
        z_filter_down = data[:,2] > lidar_height_down  # kitti 高度过滤
        filt = np.logical_and(z_filter_down, z_filter)  # 必须同时成立

        first_line_filtered = data[filt,:]
        print('first_line_filtered number.' ,first_line_filtered.shape,data.shape)
    else:
        first_line_filtered = data

    if data.shape[0] < 1900 or first_line_filtered.shape[0] < 180:
        print(' RANSAC point number too small.')
        return None, None, None, None

    L_data = data.shape[0]
    R_L = range(first_line_filtered.shape[0])


    while i < K:
        # 随机选3个点  np.random.choice 很耗费时间，改为random模块
        # s3 = np.random.choice(L_data, sample_size, replace=False)
        s3 = random.sample(R_L, sample_size)

        # 计算平面方程系数
        coeffs = estimate_plane(first_line_filtered[s3,:])
        if coeffs is None:
            continue
        # 法向量的模, 如果系数标准化了就不需要除以法向量的模了
        r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
        # 法向量与Z轴(0,0,1)夹角
        alphaz = math.acos(abs(coeffs[2]) / r)

        # r = math.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
        # 计算每个点和平面的距离，根据阈值得到距离平面较近的点数量
        # d = np.abs(np.matmul(coeffs[:3], data.T) + coeffs[3]) / r
        d = np.divide(np.abs(np.matmul(coeffs[:3], data[:,:3].T) + coeffs[3]), r)
        d_filt = np.array(d < distance_threshold)
        d_filt_object = ~d_filt

        near_point_num = np.sum(d_filt,axis=0)

        # 为了避免将平直墙面检测为地面，必须将夹角加入判断条件，
        # 如果只用内点数量就会导致某个点数较多的非地面平面占据max_point_num，
        # 而其他地面平面无法做出更换；问题是该阈值的选取多少合适？
        if near_point_num > max_point_num and alphaz < alpha_threshold:
            max_point_num = near_point_num

            best_model = coeffs
            best_filt = d_filt
            best_filt_object = d_filt_object

            # # 与Z轴(0,0,1)夹角
            # alpha = math.acos(abs(coeffs[2]) / r)
            alpha = alphaz

            w = near_point_num / L_data

            wn = math.pow(w, 3)
            p_no_outliers = 1.0 - wn
            # sd_w = np.sqrt(p_no_outliers) / wn
            K = (math.log(1-P) / math.log(p_no_outliers)) #+ sd_w

        i += 1


        if i > max_iterations:
            print(' RANSAC reached the maximum number of trials.')
            return None,None,None,None

    # print('took iterations:', i+1, 'best model:', best_model,
          # 'explains:', max_point_num)
    # 返回地面坐标，障碍物坐标，模型系数，夹角
    return np.argwhere(best_filt).flatten(),np.argwhere(best_filt_object).flatten(), best_model, alpha
  
def estimate_plane(xyz, normalize=False):
    """
    已知三个点，根据法向量求平面方程；
    返回平面方程的一般形式的四个系数
    :param xyz:  3*3 array
    x1 y1 z1
    x2 y2 z2
    x3 y3 z3
    :return: a b c d

      model_coefficients.resize (4);
      model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
      model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
      model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
      model_coefficients[3] = 0;
      model_coefficients.normalize ();
      model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot (p0.matrix ()));

    """
    vector1 = xyz[1,:] - xyz[0,:]
    vector2 = xyz[2,:] - xyz[0,:]

    # 共线性检查 ; 0过滤
    if not np.all(vector1):
        # print('will divide by zero..', vector1)
        return None
    dy1dy2 = vector2 / vector1
    # 2向量如果是一条直线，那么必然它的xyz都是同一个比例关系
    if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):
        return None


    a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
    b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
    c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])
    # normalize
    if normalize:
        # r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        r = math.sqrt(a ** 2 + b ** 2 + c ** 2)
        a = a / r
        b = b / r
        c = c / r
    d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
    # return a,b,c,d
    return np.array([a,b,c,d])

def get_boundary(cloud,dist_thold=12,vertex_num=360):
    '''
    Input parameters:
      cloud: input cloud (np array)
      dist_thold: max boundary distance
      vertex_num: number of the vertex of the boundary
    Output: 
      boundary: boundary of the collision-free region 
    '''
    boundary = [dist_thold for i in range(vertex_num)]

    for point in cloud:
      dist = math.sqrt(point[0] * point[0] + point[1] * point[1])
      theta = math.atan2(point[1],point[0])
      
      if dist > dist_thold or dist < 3:
        continue
      
      if theta < 0:
        theta = theta + 2 * np.pi

      theta_floor = int(np.floor(theta/np.pi*vertex_num/2))
      theta_ceil = theta_floor+1
      
      if theta_floor < 0 or theta_floor >= vertex_num:
        continue
      if theta_ceil == vertex_num:
        continue
      
      if boundary[theta_floor] > dist:
        boundary[theta_floor] = dist
        boundary[theta_ceil] = dist

    for j in range(2):
      for i in range(vertex_num-1):
        if boundary[i] == dist_thold and (boundary[i-1]<dist_thold and boundary[i+1]<dist_thold):
          boundary[i] = boundary[i-1]
    return boundary

def get_lane_boundary(map,trans,dist_thre=12,vertex_num=360):
    '''
    Input parameters:
        map: carla map in OPENDRIVE format
        trans: current transform of the ego vehicle 
        dist_thre: max boundary distance
        vertex_num: number of the vertex of the boundary
    Output:
        boundary: boundary of the drivable area
    '''
    
    def get_single_side(p,direction):
        '''
        Input parameters:
            p: refernce waypoint
            direction: 1 for left and -1 for right
        Output:
            single_side_boundary: location list of the boundary of the lane
        '''
        # print("s of the point: {:.2f}".format(p.s))
        # print("Previous wp: {}, next wp: {}".format(p.previous(2.0),p.next(2.0)))
        # if p.s <= 0.2:
        #     prev_wps = [p]
        # else:
        #     prev_wps = p.previous_until_lane_start(0.1)
        # prev_wps = p.previous_until_lane_start(0.1)    
        # print("len of prev wps = {}".format(len(prev_wps)))
        # next_wps = p.next_until_lane_end(0.1)
        # print("len of next wps = {}".format(len(next_wps)))
        prev_wps = []
        next_wps = []

        for i in range(1,100):
            # if p.s > i*0.2:
            prev_wp = p.previous(i*0.2)
            if len(prev_wp) > 0:
                prev_wps.append(prev_wp[0])
            next_wp = p.next(i*0.2)
            if len(next_wp) > 0:
                next_wps.append(next_wp[0])
        # print("prev_wps id: {}, next_wps id: {}.".format([w.lane_id for w in prev_wps],[w.lane_id for w in next_wps]))
        # print("prev_wps len: {}, next_wps len: {}.".format(len(prev_wps),len(next_wps)))
        lane_wps = prev_wps + next_wps
        # if no lane waypoints 
        if len(lane_wps) <= 1: 
            print("waypoints are too few.")
            return []
        single_side_boundary = [lateral_shift(w.transform, -direction * w.lane_width*0.5) for w in lane_wps]
        return single_side_boundary

    def lateral_shift(transform, shift):
        '''
        Input parameters:
            transform: transform of the waypoint
            shift: lateral shift distance for the waypoint
        Output:
            shifted_loc: the waypoint's location after the shift 
        '''
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def get_locs_in_boundary(wps_loc,ego_loc,thre):
        for loc in wps_loc:
            if (loc.x - ego_loc.x)**2 + (loc.y - ego_loc.y)**2 > thre**2:
                wps_loc.remove(loc)
        if len(wps_loc) <= 1: 
            print("waypoints are too few after filtering with distance threshold.")
            return []
        return wps_loc

    ## get continuous boundary for each side
    def get_continous_boundary(origin_boundary,trans,dist_thre,vertex_num):
        '''
        Input parameters: 
            origin_boundary: origin boundary in ordinate
            trans: transform of the ego vehicle at current time
            dist_thre
            vertex_num
        Output:
            continous_boundary: continous boundary in solar system list([angle,dist])
        '''
        side_polar = []
        # convert the side coordinate into polar system
        for loc in origin_boundary:
            yaw = trans.rotation.yaw/180*np.pi
            x = loc.x
            y = loc.y
            x0 = trans.location.x
            y0 = trans.location.y
            dist = math.sqrt(((x-x0) ** 2 + (y-y0) ** 2))
            theta = -math.atan2((x-x0)*math.cos(yaw)+(y-y0)*math.sin(yaw),-(x-x0)*math.sin(yaw)+(y-y0)*math.cos(yaw))
            if dist > dist_thre:
                continue
            if theta < 0:
                theta = theta + 2 * np.pi
            theta_floor = int(np.floor(theta/np.pi*vertex_num/2))
            if theta_floor >= 0 and theta_floor <= 90:
                theta_floor += 360
            side_polar.append([theta_floor,dist])
        
        # add boundary point that is not continous
        side_polar.sort()
        
        continous_boundary = side_polar
        min_theta = min(side_polar)[0]
        max_theta = max(side_polar)[0]
        angle_list = [p[0] for p in continous_boundary]
        # print("angle range: max = {}, min = {}".format(max_theta,min_theta))
        for angle in range(min_theta,max_theta):
            if angle in angle_list:
                continue
            else:
                # this angle does not exist in the side_solar, need to add
                idx = angle_list.index(angle-1)
                continous_boundary.append([angle,continous_boundary[idx][1]])
                angle_list.append(angle)    
        
        continous_boundary.sort()
        return continous_boundary


    # Get reference waypoint
    ref_wp = map.get_waypoint(trans.location,project_to_road=True,
                            lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    boundary = [dist_thre for i in range(vertex_num)]
    
    if ref_wp is None:
        print("No reference waypoint found.")
        return boundary
    ## check if the current location waypoint is in the driving lane 
    if ref_wp.lane_type is not carla.LaneType.Driving:
        # print("The ego vehicle is not in driving lane")
        return boundary

    ## check whether in junction
    in_junction = ref_wp.is_junction

    if in_junction:
        # print("The reference waypoint is in junction.")
        return boundary
        ### TODO: deal with the situations in junction
    else:
        ### if in lane, get lane change, and for each lane change case
        ### get the left and the right lane boundary
        lane_change = ref_wp.lane_change
        # print("LaneChange of current wp is {}.".format(lane_change))
        left_side = []
        right_side = []
        if lane_change == carla.LaneChange.Both:
            left_wp = ref_wp.get_left_lane()
            if left_wp is None: 
                print("left waypoint is None")
                left_wp = ref_wp
            left_side = get_single_side(left_wp,1)
            right_wp = ref_wp.get_right_lane()
            if right_wp is None:
                print("right waypoint is None")
                right_wp = ref_wp
            right_side = get_single_side(right_wp,-1)
        elif lane_change == carla.LaneChange.Left:
            left_wp = ref_wp.get_left_lane()
            if left_wp is None: 
                print("left waypoint is None")
                left_wp = ref_wp
            left_side = get_single_side(left_wp,1)
            right_side = get_single_side(ref_wp,-1)
        elif lane_change == carla.LaneChange.Right:
            left_side = get_single_side(ref_wp,1)
            right_wp = ref_wp.get_right_lane()
            if right_wp is None:
                print("right waypoint is None")
                right_wp = ref_wp
            right_side = get_single_side(right_wp,-1)
        else: # LaneChange.None
            left_side = get_single_side(ref_wp,1)
            right_side = get_single_side(ref_wp,-1)

        ## get rid of the lane points out of the distance threshold
        if len(left_side) > 1: 
            left_side = get_locs_in_boundary(left_side,trans.location,dist_thre)
            left_side_polar = get_continous_boundary(left_side,trans,dist_thre,vertex_num)
        else:
            left_side_polar = []
            
        if len(right_side) > 1: 
            right_side = get_locs_in_boundary(right_side,trans.location,dist_thre)
            right_side_polar = get_continous_boundary(right_side,trans,dist_thre,vertex_num)
        else:
            right_side_polar = []
        both_sides = left_side + right_side

        ## calculate the lane boundart in the polar system
        if len(both_sides)<=1: return boundary
        
        for boundary_point in left_side_polar+right_side_polar:
            th = boundary_point[0]
            if th >= 360: th -= 360
            di = boundary_point[1]
            if boundary[th] > di:
                boundary[th] = di

    return boundary
how to use:

S2Track S2track;
S2track = S2Track (input_viewer, input_viewID);
S2track.update (GlobalVariable::DEBUG_MODE, cur_cluster, cur_cluster_num);
cout << "[S2track           ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;
    
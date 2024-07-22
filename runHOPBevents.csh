# shell script to run different tagged events
# Two restart files because Huancui was playing with 2 simulations. One with and without routing!  
set path_rst1 = "/glade/derecho/scratch/butlerz/FROM_CHEYENNE/wrf_running_NEON/wrf_v5.0/HOPB"
#set path_rst2 = "/glade/scratch/butlerz/wrf_running_NEON/default_run/KING/"
set filestart = "/glade/derecho/scratch/butlerz/FROM_CHEYENNE/tracer_NEON/wrf_v5.0/calib/HOPB/Start_tracer_dates.txt"
set fileend   = "/glade/derecho/scratch/butlerz/FROM_CHEYENNE/tracer_NEON/wrf_v5.0/calib/HOPB/End_tracer_dates.txt"
#set fileWRF   = "/glade/scratch/butlerz/tracer_NEON/calib/BLUE/Start_WRF_dates.txt"

# Read in files above
set events    =  `cat $filestart`
set eventends =  `cat $fileend`
#set wrf_event =  `cat $fileWRF`
set tag_ref    =  $events[2]
set tagend_ref =  $eventends[2]
#set eventwrf_ref = $wrf_event[1]

# What does this i do?
set i = 1 #where to start the tagging...  #start tagging events (1). 
# Number of events below
while ( $i <= 84 ) 
    echo 'first line in the loop'
    @ j = $i + 1
    echo $j
    echo 'third line in the loop'
    echo $events[$i]
    echo $events[$j]
    if ($i >= 1) then
      set simu_year  = `echo $events[$i]|cut -c1-4`
      set simu_month = `echo $events[$i]|cut -c5-6`
      set tag_year   = `echo $events[$j]|cut -c1-4`
      set tag_month  = `echo $events[$j]|cut -c5-6`
      echo $simu_year
      echo $simu_month
      echo $tag_year
      echo $tag_month
     
      # Name folders based on iterations of tracer events. 001, 010, 100 for example
      if ($i < 10) then
        set event_id = "tag00"${i}
      else
        if ($i < 100) then
          set event_id = "tag0"${i}
        else
          set event_id = "tag"${i}
        endif
      endif
      echo $event_id 
      mkdir ${event_id}
      #mkdir ${event_id}/channel_gw
      #mkdir ${event_id}/channel_gw_ovrt_subrt
    
      # Link and Copy hhuancui directories  
      ln -sf /glade/work/hhuancui/wrf_hydro_nwm_public/trunk/NDHMS/Run/*.exe $event_id/
      cp /glade/work/hhuancui/wrf_hydro_nwm_public/trunk/NDHMS/Run/*.TBL $event_id/
      #ln -sf /glade/scratch/butlerz/wrf_running_NEON/BLUE/TEST/*.exe $event_id/
      #cp  /glaade/scratch/butlerz/wrf_running_NEON/BLUE/TEST/*.TBL $event_id/

      # Two namelist's and running script
      cp /glade/derecho/scratch/butlerz/FROM_CHEYENNE/tracer_NEON/wrf_v5.0/calib/HOPB/tag001/namelist.hrldas $event_id/
      cp /glade/derecho/scratch/butlerz/FROM_CHEYENNE/tracer_NEON/wrf_v5.0/calib/HOPB/tag001/hydro.namelist $event_id/
      cp /glade/derecho/scratch/butlerz/FROM_CHEYENNE/tracer_NEON/wrf_v5.0/calib/HOPB/tag001/WRF_HOPB.sh $event_id/
      cd $event_id/
      perl -pi -e 's/'${tag_ref}'/'$events[$j]'/g' namelist.hrldas
      perl -pi -e 's/'${tagend_ref}'/'$eventends[$j]'/g' namelist.hrldas
      perl -pi -e 's/tag001/'$event_id'/g' namelist.hrldas # Controls OUTDIR in namelist to go to each new event
      perl -pi -e 's/START_YEAR  = 2015/START_YEAR  = '$simu_year'/g' namelist.hrldas
      perl -pi -e 's/START_MONTH  = 12/START_MONTH  = '$simu_month'/g' namelist.hrldas
      perl -pi -e 's/RESTART.2015120100/RESTART.'$simu_year$simu_month'0100/g' namelist.hrldas
      perl -pi -e 's/HYDRO_RST.2015-12-01/HYDRO_RST.'$simu_year'-'$simu_month'-01/g' hydro.namelist
      cp $path_rst1/RESTART.${simu_year}${simu_month}0100_DOMAIN1 .
      #cp $path_rst1/RESTART.2016010100_DOMAIN1 .
      
      cp $path_rst1/HYDRO_RST.${simu_year}-${simu_month}-01_00:00_DOMAIN1 .
      qsub WRF_HOPB.sh
      cd ../

      # Do not need below. This is when Huancui wwas playing with routing  
      #ln -sf /glade/work/hhuancui/wrf_hydro_nwm_public/trunk/NDHMS/Run/*.exe $event_id/channel_gw_ovrt_subrt/
      #cp /glade/work/hhuancui/wrf_hydro_nwm_public/trunk/NDHMS/Run/*.TBL $event_id/channel_gw_ovrt_subrt/
      #cp /glade/scratch/butlerz/tracer_NEON/default_run/KING/tag001/channel_gw_ovrt_subrt/namelist.hrldas $event_id/channel_gw_ovrt_subrt/
      #cp /glade/scratch/butlerz/tracer_NEON/default_run/KING/tag001/channel_gw_ovrt_subrt/hydro.namelist $event_id/channel_gw_ovrt_subrt/
      #cp /glade/scratch/butlerz/tracer_NEON/default_run/KING/tag001/channel_gw_ovrt_subrt/WRF_KING.sh $event_id/channel_gw_ovrt_subrt/
      #cd $event_id/channel_gw_ovrt_subrt/
      #perl -pi -e 's/'${event_ref}'/'$events[$i]'/g' namelist.hrldas
      #perl -pi -e 's/'${eventend_ref}'/'$eventends[$i]'/g' namelist.hrldas
      #perl -pi -e 's/tag001/'$event_id'/g' namelist.hrldas
      #perl -pi -e 's/START_YEAR  = 2016/START_YEAR  = '$event_year'/g' namelist.hrldas
      #perl -pi -e 's/START_MONTH = 01/START_MONTH  = '$event_month'/g' namelist.hrldas
      #perl -pi -e 's/RESTART.2016010100/RESTART.'$event_year$event_month'0100/g' namelist.hrldas 
      #perl -pi -e 's/HYDRO_RST.2016-01-01/HYDRO_RST.'$event_year'-'$event_month'-01/g' hydro.namelist
      #cp $path_rst2/RESTART.${event_year}${event_month}0100_DOMAIN1 .
      #cp $path_rst2/HYDRO_RST.${event_year}-${event_month}-01_00:00_DOMAIN1 .
      #qsub WRF_KING.sh
      #cd ../../

    endif
    @ i = $i + 1
end

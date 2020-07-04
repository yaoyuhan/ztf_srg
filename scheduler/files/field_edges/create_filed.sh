for i in $(seq 200 879);
    do echo $i
    /home/tb/ztfpt $i out=$i.dat
done

for i in $(seq 1001 1773);
    do echo $i
    /home/tb/ztfpt $i out=$i.dat
done


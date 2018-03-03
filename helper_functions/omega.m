function O = omega(v)
    O = [0 -v(1) -v(2) -v(3);
        v(1) 0 v(3) -v(2);
        v(2) -v(3) 0 v(1);
        v(3) v(2) -v(1) 0];
end
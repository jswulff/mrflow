all:
	rm -f eff_trws.so
	/usr/bin/env python2 setup.py build_ext --inplace
	ln -sf `find . -name eff_trws.so` .
	#ln -sf ${PWD}/eff_trws.so ..

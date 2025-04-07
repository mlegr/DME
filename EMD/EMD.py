# -*- coding: utf-8 -*-
from __future__ import division
import scipy.io
import matplotlib as plt
import math
import time
import numpy as np
from pylab import *
from scipy.interpolate import interp1d,spline,splrep,splev
from scipy.signal import hilbert
import os
from mpl_toolkits.mplot3d import Axes3D

def find_extrema(signal):
	min_loc = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0] + 1 # Min local 
	max_loc = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1 # Max local 
	ext_loc = np.diff(np.sign(np.diff(signal))).nonzero()[0] + 1 		 # Min et Max local
	return min_loc,max_loc,ext_loc

def def_enveloppe(minima,maxima,temps,signal,choix_env):
	if choix_env==Spline:
		# Methode 1 avec splrep/splev 
		interp_sup=splrep(temps[maxima],signal[maxima])
		env_sup=splev(temps,interp_sup)
		interp_inf=splrep(temps[minima],signal[minima])
		env_inf=splev(temps,interp_inf)
	elif choix_env==Hermite:
		# Methode 2 par PchipInterpolator
		f=scipy.interpolate.PchipInterpolator(temps[maxima],x[maxima])
		env_sup=f(temps)
		g=scipy.interpolate.PchipInterpolator(temps[minima],x[minima])
		env_inf=g(temps)
	return env_inf,env_sup

def sifting_process(signal,temps):
	boundary=4
	minima,maxima,extrema=find_extrema(signal)

	# prolongement des extremas s1: Deux maxima rajoutes par symetrie miroir par rapport au bord (idem minima)
	# prolongement gauche abscisse maxima
	abscisse=temps[maxima]
	abs_moins_1= -temps[maxima][0]
	abs_moins_2= -temps[maxima][1]
	abscisse=np.concatenate(([abs_moins_2],[abs_moins_1],abscisse))
	# prolongement gauche ordonnee maxima
	ordonnee=signal[maxima]
	ord_moins_1= signal[maxima][0]
	ord_moins_2= signal[maxima][1]
	ordonnee=np.concatenate(([ord_moins_2],[ord_moins_1],ordonnee))
	# prolongement droit abscisse  maxima
	abs_plus_1= temps[-1] +(temps[-1]-temps[maxima][-1])
	abs_plus_2= temps[-1] +(temps[-1]-temps[maxima][-2])
	abscisse=np.concatenate((abscisse,[abs_plus_1],[abs_plus_2]))
	# prolongement droit ordonnee maxima
	ord_plus_1= signal[maxima][-1]
	ord_plus_2= signal[maxima][-2]
	ordonnee=np.concatenate((ordonnee,[ord_plus_1],[ord_plus_2]))
	# generation de l'enveloppe sup
	interp_sup=splrep(abscisse,ordonnee)
	env_sup=splev(temps,interp_sup)
	# prolongement gauche abscisse minima
	abscisse=temps[minima]
	abs_moins_1= -temps[minima][0]
	abs_moins_2= -temps[minima][1]
	abscisse=np.concatenate(([abs_moins_2],[abs_moins_1],abscisse))
	# prolongement gauche ordonnee minima
	ordonnee=signal[minima]
	ord_moins_1= signal[minima][0]
	ord_moins_2= signal[minima][1]
	ordonnee=np.concatenate(([ord_moins_2],[ord_moins_1],ordonnee))
	# prolongement droit abscisse  minima
	abs_plus_1= temps[-1] +(temps[-1]-temps[minima][-1])
	abs_plus_2= temps[-1] +(temps[-1]-temps[minima][-2])
	abscisse=np.concatenate((abscisse,[abs_plus_1],[abs_plus_2]))
	# prolongement droit ordonnee minima
	ord_plus_1= signal[minima][-1]
	ord_plus_2= signal[minima][-2]
	ordonnee=np.concatenate((ordonnee,[ord_plus_1],[ord_plus_2]))
	#Generation de l'enveloppe inf
	interp_inf=splrep(abscisse,ordonnee)
	env_inf=splev(temps,interp_inf)
	#Moyenne
	moyenne=(env_inf+env_sup)/2
	return moyenne

def calcul_SD(ancien_h,h,temps):
	SD=(np.linalg.norm(abs(ancien_h-h))/np.linalg.norm(abs(ancien_h)))**2
	return SD

def movingaverage_bord(signal, window_size):
	'''Calcul le moving average du signal en tenant compte des bords'''
	window= np.ones(int(window_size))/float(window_size)
	signal= np.convolve(signal, window,'same')
	nb_gauche=int(np.floor((window_size-1)/2))
	nb_droite=int((window_size-1)-nb_gauche)
	for i in range(0,nb_droite):
		signal[i]=signal[i]*window_size/(window_size- nb_droite+i)
	for i in range(0,nb_gauche):
		signal[len(signal)-nb_gauche+i]=signal[len(signal)-nb_gauche+i]*window_size/(window_size- 1 -i)
	return signal

def test_IMF(signal):
	ext_loc = np.diff(np.sign(np.diff(signal))).nonzero()[0] + 1 		 # Extrema locaux
	nb_ext=len(ext_loc)
	if nb_ext<3:
		bool_residu=True
	else:
		bool_residu=False
	return bool_residu

def moyenne(signal,poids):
	n=len(signal)
	nv=np.floor(n/poids)
	sv=np.zeros(nv)
	for i in range(0,len(sv)-1):
		sv[i]=np.mean(signal[i*poids:(i+1)*poids])
	return sv

def visu(imf):
	fig=figure ('PF ',figsize=(15,10))
	for i in range (1,min(imf["nb_imf"]+1,6)):
		fig.add_subplot(min(imf["nb_imf"],5),1,i)
		title("IMF"+str(i))
		plot(imf["temps"],imf["IMF"]["IMF"+str(i)])
		ylim([min(imf["signal"]),max(imf["signal"])])
	reconstitution(imf)
	show()

def reconstitution(imf):
	figure('signal initiale en bleu/ signal recompose en rouge/ residu en vert')
	plot(imf["temps"],imf["signal"],'b')
	i=imf["nb_imf"]
	sommes_IMF=np.zeros(len(imf["temps"]))
	while i>0:
		sommes_IMF=sommes_IMF+imf["IMF"]["IMF"+str(i)]
		i-=1
	sommes_IMF=sommes_IMF+imf["residu"]
	plot(imf["temps"],sommes_IMF,'r')
	plot(imf["temps"],imf["residu"],'g')
	
def verif_criteres_IMF(imf):
	''' Verifie si les IMF calculees remplissent les criteres definies par Huang:
			1- sur l'ensemble de la fonction, le nombre d'extrema et le nombre d'intersections avec y=0 doit etre egal ou differe au plus de 1.
			2- en tout point, la valeur moyenne de l'enveloppe definie par les maxima locaux et de l'enveloppe definie par les minima locaux vaut 0.
	'''
	non_IMF=[]
	i=imf["nb_imf"]
	while i>0:
		#Calcul du nombre d' extrema de l'IMF
		nb_extrema=len(np.diff(np.sign(np.diff(imf["IMF"]["IMF"+str(i)]))).nonzero()[0] + 1)
		#Calcul du nombre de zeros de l'IMF
		nb_zeros=0
		for j in range (0,len(imf["IMF"]["IMF"+str(i)])-2):
			if imf["IMF"]["IMF"+str(i)][j]*imf["IMF"]["IMF"+str(i)][j+1]<0:
				nb_zeros+=1
		i-=1
		if abs(nb_zeros- nb_extrema)>1:
			non_IMF=np.concatenate([non_IMF,[i]])

	if len(non_IMF)==0:
		print 'Toutes les IMF verifient les criteres de Huang'
	else:
		nb_non_IMF=len(non_IMF)
		print 'Les IMF:'
		while nb_non_IMF>0:
			print nb_non_IMF
			nb_non_IMF-=1
		print 'ne verifient pas les criteres de Huang'

def sauvegarde(imf,temps):
	'''Stockage des donnees'''
	i=1
	print 'fichier LaTeX'
	fichier = open('signal.tex','w')
	fichier.write('\\addplot[thick,color=vertfonce,line join=round] coordinates{\n')
	for j in range (0,len(temps)):
		fichier.write('('+str("%.4f" % temps[j])+','+str("%.3f" % imf['signal'][j])+')\n')
	fichier.write('};\n')
	while 'IMF'+str(i) in imf["IMF"].keys():
		print 'traitement IMF'
		print '#########'
		fichier = open('IMF'+str(i)+'.tex','w')
		fichier.write('\\addplot[thick,color=vertfonce,line join=round] coordinates{\n')
		for j in range (0,len(temps)):
			fichier.write('('+str("%.4f" % temps[j])+','+str("%.3f" % imf['IMF']["IMF"+str(i)][j])+')\n')
		fichier.write('};\n')
		i=i+1

def trace_pdf(temps,x,imf):
	fichier = open('x.tex','w')
	fichier.write('\\documentclass[12pt]{standalone}\n')
	fichier.write('\\usepackage[x11names]{xcolor}\n')
	fichier.write('\\usepackage{tikz}\n')
	fichier.write('\\usepackage{pgfplots}\n')
	fichier.write('\\definecolor{vertfonce}{RGB}{0,95,19}\n')
	fichier.write('%\n')
	fichier.write('\\begin{document}\n')
	fichier.write('\\small\n')
	fichier.write('\\begin{tikzpicture}\n')
	fichier.write('\\begin{axis}[\n')
	fichier.write('xmin='+str(temps[0])+',\n')
	fichier.write('xmax='+str(temps[-1])+',\n')
	fichier.write('ymin='+str(min(x))+',\n')
	fichier.write('ymax='+str(max(x))+',\n')
	#fichier.write('xlabel=temps (s),\n')
	fichier.write('axis background/.style={fill=black!5},\n')
	fichier.write('ylabel=$x(t)$ ,\n')
	#fichier.write('scale only axis,\n')
	fichier.write('scaled x ticks = false,\n')
	fichier.write('scaled y ticks = false,\n')
	fichier.write('each nth point={10},\n')
	fichier.write('filter discard warning=false,\n')	
	fichier.write('xticklabels={,,},\n')
	fichier.write('width=9.3cm,\n')
	fichier.write('height=4cm,\n')
	fichier.write('axis on top\n')
	fichier.write(']\n')
	fichier.write('\\input{signal}\n')
	fichier.write('\\end{axis}\n')
	fichier.write('\\end{tikzpicture}\n')
	fichier.write('\\end{document}\n')
	fichier.close()
	# Compilation
	compil = 'lualatex'
	os.system(compil+' x.tex')	
	i=1
	while 'IMF'+str(i) in imf["IMF"].keys():
		fichier = open('gamma'+str(i)+'.tex','w')
		fichier.write('\\documentclass[12pt]{standalone}\n')
		fichier.write('\\usepackage[x11names]{xcolor}\n')
		fichier.write('\\usepackage{tikz}\n')
		fichier.write('\\usepackage{pgfplots}\n')
		fichier.write('\\definecolor{vertfonce}{RGB}{0,95,19}\n')
		fichier.write('%\n')
		fichier.write('\\begin{document}\n')
		fichier.write('\\small\n')
		fichier.write('\\begin{tikzpicture}\n')
		fichier.write('\\begin{axis}[\n')
		fichier.write('xmin='+str(temps[0])+',\n')
		fichier.write('xmax='+str(temps[-1])+',\n')
		fichier.write('ymin='+str(min(x))+',\n')
		fichier.write('ymax='+str(max(x))+',\n')
		#fichier.write('xlabel=temps (s),\n')
		fichier.write('axis background/.style={fill=black!5},\n')
		fichier.write('ylabel=$\\gamma_'+str(int(i))+'(t)$ ,\n')
		#fichier.write('scale only axis,\n')
		fichier.write('scaled x ticks = false,\n')
		fichier.write('scaled y ticks = false,\n')
		fichier.write('each nth point={10},\n')
		fichier.write('filter discard warning=false,\n')	
		fichier.write('xticklabels={,,},\n')
		fichier.write('width=9.3cm,\n')
		fichier.write('height=4cm,\n')
		fichier.write('axis on top\n')
		fichier.write(']\n')
		fichier.write('\\input{IMF'+str(int(i))+'}\n')
		fichier.write('\\end{axis}\n')
		fichier.write('\\end{tikzpicture}\n')
		fichier.write('\\end{document}\n')
		fichier.close()
		# Compilation
		compil = 'lualatex'
		os.system(compil+' gamma'+str(i)+'.tex')
		i+=1

def init():
	affichage = 2
	itermax=2000 							# parametres (nb cycles = nb. max d'IMF calculees et iterations = nombre d'iterations max pour 1 IMF)
	nb_cycles_max=20
	crit_conv = 1e-12  					# Critere de convergence pour l'obtention d'une IMF
	crit_conv_SD = 0.3					# Critere de convergence pour l'obtention d'une IMF standard deviation
	check_res = 1							# pour chaque IMF, verification validite (0->non, 1->oui)
	signal = 3									# choix du signal

	if signal == 1:
		# signal 1:
		temps=np.linspace(0,10,10001)
		x=np.sin(2*pi*temps)+np.cos(3*2*pi*temps)+(np.sin(2*pi*temps))**2.*np.sin(3*2*pi*temps)
	elif signal == 2:
		# signal 3:
		t=np.linspace(0,5*np.pi,15709)
		temps=t
		x=t**2.*np.sin(t**2)
	elif signal == 3:
		# signal 6:
		t=np.linspace(0,8*np.pi,25134)
		temps=t
		x=np.cos(t**2)+3*np.cos(3*t)+np.cos(t) 
	elif signal == 4:
		# signal 7:
		temps = np.linspace(0,20*np.pi,100000)
		x = (np.exp(-0.2*temps)*temps**2+10)*np.sin(np.exp(-0.1*temps)*temps**2+3*temps)
		#ajout composante HF
		x = x + np.sin(150*temps) + np.cos(74*temps) 
	elif signal == 5:
		# signal 8:
		temps=np.linspace(0,30*np.pi,9426)
		x = (np.exp(-0.2*temps)*temps**2+10)*(np.sin(np.exp(-0.1*temps)*temps**2+3*temps)+np.cos(2*temps)) 
	elif signal == 6:
		# signal 9:
		temps=np.linspace(0,30*np.pi,9426)
		x = np.cos(5*temps)+np.cos(2*temps) 
	elif signal == 7:
		# signal 10:
		temps=np.linspace(0,1,10000)
		x=(1+0.5*np.sin(10*pi*temps))*np.cos(100*pi*temps+2*np.sin(14*pi*temps))+np.sin(20*pi*temps)
	plot(temps,x)
	show()
	return x,temps,affichage,itermax,nb_cycles_max,crit_conv,crit_conv_SD,check_res

	
def emd():
	#initialisation des variables du processus
	x,temps,affichage,itermax,nb_cycles_max,crit_conv,crit_conv_SD,check_res=init()
	r=x
	bool_residu=False
	nb_cycle=1
	imf={}
	imf["temps"]=temps
	imf["opt_degen"]=True
	imf["signal"]=x
	imf["IMF"]={}
	imf["AMP"]={}
	imf["IF"]={}
	#Sifting process
	while bool_residu==False and nb_cycle< nb_cycles_max:
		h=r
		SD=crit_conv_SD+1
		iteration=1
		print 'calcul de l IMF',nb_cycle
		while (SD>crit_conv_SD) and (iteration<itermax):
			ancien_h=h
			# Dans le cas ou on utilise la condition de bord 4 le signal doit posseder au moins 4 extrema.
			nb_ext_h=len(np.diff(np.sign(np.diff(ancien_h))).nonzero()[0] + 1)
			if nb_ext_h<4:
				break
			#Calcul de la moyenne
			m=sifting_process(ancien_h,temps)
			#Calcul du nouveau h(t)
			h=ancien_h-m
			#Calcul de la standard deviation SD
			SD=calcul_SD(ancien_h,h,temps)
			print 'iteration ',iteration ,'SD=',SD
			iteration=iteration+1

		# Soustraction de l'IMF
		r=r-h
		# Stockage de l'IMF
		imf["IMF"]["IMF"+str(nb_cycle)]=h
		# Verification du nombre d'extrema du residu
		bool_residu=test_IMF(r)
		nb_cycle=nb_cycle+1
	#Stockage du nombre d'IMF et du résidu
	imf["nb_imf"]=nb_cycle-1
	imf["residu"]=r
	# Verification des IMF selon Huang
	verif_criteres_IMF(imf)
	#Visualisation des IMF
	visu(imf)

	
	fold = 0	
	crea = 0
	while crea == 0:
	    try:
	        os.mkdir('./res')
	        os.chdir('./res')
	        crea = 1
	    except:
	        fold += 1
	        try:
	            os.mkdir('./res_'+str(fold))
	            os.chdir('./res')
	            crea = 1    
	        except:
	            '.'
	
	scipy.io.savemat("imf.mat",imf)
	sauvegarde(imf,temps)
	trace_pdf(temps,x,imf)
	Hilbert_pdf(imf)
	return imf

imf=emd()

	

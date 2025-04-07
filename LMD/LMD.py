# -*- coding: utf-8 -*-
from __future__ import division
import scipy.io
import matplotlib as plt
import math
import time
import os
import numpy as np
from pylab import *
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d,spline,splrep,splev

def movingaverage_bord(signal, window_size):
	window= np.ones(int(window_size))/float(window_size)
	signal= np.convolve(signal, window,'same')
	nb_gauche=int(np.floor((window_size-1)/2))
	nb_droite=int((window_size-1)-nb_gauche)
	for i in range(0,nb_droite):
		signal[i]=signal[i]*window_size/(window_size- nb_droite+i)
	for i in range(0,nb_gauche):
		signal[len(signal)-nb_gauche+i]=signal[len(signal)-nb_gauche+i]*window_size/(window_size- 1 -i)
	return signal

def smoothing(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')	
	# source: http://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
	
def cond_lim_1 (signal,abscisse_x,abscisse_extrema, extrema):
	# Cette fonction definie des conditions limites en considerant les valeurs initiales et finales comme etant des extrema de la fonction
	m_init=(signal[0]+extrema[0])/2
	a_init=abs(signal[0]-extrema[0])/2
	m_fin=(signal[len(abscisse_x)-1]+ extrema[len(abscisse_extrema)-1])/2
	a_fin=abs(signal[len(abscisse_x)-1]- extrema[len(abscisse_extrema)-1])/2
	return m_init,m_fin,a_init,a_fin

def cond_lim_2(mi,ai):
	# Cette fonction definie des conditions limites en considerant les valeurs initiales (resp. finales) comme etant les valeurs du 1er (resp. dernier) mi ou ai de la fonction
	m_init=mi[0]
	a_init=ai[0]
	m_fin=mi[len(mi)-1]
	a_fin=ai[len(ai)-1]
	return m_init,m_fin,a_init,a_fin

def cherche_extremas_locaux (signal):
	b = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0] + 1 # Min local 
	c = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1 # Max local 
	d = np.diff(np.sign(np.diff(signal))).nonzero()[0] + 1 		 # Min et Max local
	return b,c,d

def calcul_mi_ai(extrema):
	mi = np.zeros([len(extrema)-1])
	ai = np.zeros([len(extrema)-1])
	for i in range(0,len(mi)): 
		mi[i]= (extrema[i]+extrema[i+1])/2
		ai[i]=abs(extrema[i]-extrema[i+1])/2
	return mi,ai

def calcul_SD(ancien_h,h):
	SD=(np.linalg.norm(abs(ancien_h-h))/np.linalg.norm(ancien_h))**2
	return SD

def calcul_IMF_IF(enveloppe_finale,signal,sum_IMF,pas):
	# fonction produit			
	IMF = np.zeros([len(signal)])
	IMF=np.multiply(enveloppe_finale,signal)
	sum_IMF=sum_IMF+IMF
	# phase
	phi=np.arccos(signal) 
	# frequence instantanee
	omega=abs(np.diff(phi)/pas) 
	return IMF,sum_IMF,phi,omega

def ext(signal,facteur):
	# differentes façons d'etendre le signal...
	
	# ... par symetrie centrale par rapport aux points extremes
	long = len(signal)
	p = int(np.floor(long*facteur))
	gauche = np.zeros([p])
	droite = np.zeros([p])
	for i in range(p):
		gauche[i] = signal[0]+(signal[0]-signal[i+1])
		droite[i] = signal[-1]+(signal[-1]-signal[-1-i-1])
	gauche = gauche[::-1]
	signal_ext = np.concatenate((gauche,signal,droite))
	return signal_ext,p
	
def rest(signal,p,nb_points):
	long = len(signal)
	inter = np.floor((long-nb_points)/2.)
	signal = signal[inter:len(signal)-inter]
	return signal
	
def lmd(x_extrema,s,nb_points,iter_ma,facteur):
	choix=1															# choix de la methode utulisee lors de chaque iteration
	extrema=s[x_extrema]									# valeurs des extrema
	[mi,ai]=calcul_mi_ai(extrema)							# calcul des mi et des ai
	# calcul sans rallongement mais avec conditions limites puis à l'aide d'un moving average successif (<iter_ma)
	# choix des conditions limites
	# Dans le cas ou le signal n'a pas assez d'extrema
	if (len(x_extrema)==1):
		test_nb_extrema=True
	else:
		test_nb_extrema=False

	if (test_nb_extrema==True):
		[m_init,m_fin,a_init,a_fin]=cond_lim_1 (s,x,x_extrema, extrema)
	else:
		[m_init,m_fin,a_init,a_fin]=cond_lim_2 (mi,ai)
	# Ajout des conditions limites
	mi = np.concatenate([[m_init],mi,[m_fin]])	
	ai = np.concatenate([[a_init],ai,[a_fin]])
	# ajout des indices de début et de fin
	x_extrema=np.concatenate(([0],x_extrema,[len(s)-1]))
	# Creation des fonctions escaliers m(t) et a(t)
	local_mean_function=np.zeros(len(s))
	local_magnitude= np.zeros(len(s))
	for i in range (0,len(ai)):
		local_mean_function[x_extrema[i]:x_extrema[i+1]]=mi[i]*np.ones(x_extrema[i+1]-x_extrema[i])
		local_magnitude[x_extrema[i]:x_extrema[i+1]]=ai[i]*np.ones(x_extrema[i+1]-x_extrema[i])
	# Correction pour éviter que le dernier terme soit nul
	local_mean_function[-1] = local_mean_function[-2]
	local_magnitude[-1] = local_magnitude[-2]
	# Taille de la fenetre selom Smith pour le moving average
	max_longueur_tiret=max(np.diff(x_extrema))
	fenetre=np.floor((max_longueur_tiret)/3)
	nb_ma=0
	smoothed_signal=local_mean_function
	for i in range(0,len(s)-1):
		if (smoothed_signal[i]==smoothed_signal[i+1]) and nb_ma<iter_ma:
			smoothed_signal = movingaverage_bord(smoothed_signal,fenetre)
			nb_ma+=1
	smoothed_mean=smoothed_signal
	nb_ma=0
	smoothed_signal=local_magnitude
	for i in range(0,len(s)-1):
		if (smoothed_signal[i]==smoothed_signal[i+1]) and nb_ma<iter_ma:
			smoothed_signal = movingaverage_bord(smoothed_signal,fenetre)
			nb_ma+=1
	smoothed_magnitude=smoothed_signal
	return smoothed_mean,smoothed_magnitude

def test_non_oscil(signal):
	ext_loc = np.diff(np.sign(np.diff(signal))).nonzero()[0] + 1 		 # Extrema locaux
	nb_ext=len(ext_loc)
	print 'nombre d extrema :',nb_ext
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

def is_monotone(signal):
	if ((sorted(signal) == signal).all())==True:
		bool_monotone=1
	elif ((sorted(-signal) == -signal).all())==True:
		bool_monotone=1
	else:
		bool_monotone=0
	return bool_monotone

def visu(imf):
	k=1
	fig=figure ('IMF et IF',figsize=(15,10))
	for i in range(1,min(imf["nb_IMF"]+1,6)):
		fig.add_subplot(min(imf["nb_IMF"],5),2,k)
		title("IMF"+str(i))
		plot(imf["temps"],imf["IMF"]["IMF"+str(i)], label='product function')
		ylim([min(imf["signal"]),max(imf["signal"])])
		legend(loc=2,prop={'size':10})
		k += 1
		fig.add_subplot(min(imf["nb_IMF"],5),2,k)
		title("IF"+str(i))
		plot(imf["temps"][0:-1],imf["IF"]["IF"+str(i)], label='instantaneous frequency')
		legend(loc=2,prop={'size':10})
		k +=1
	reconstitution(imf)
	show()

def reconstitution(imf):
	figure('Comparaison signal initial et somme des IMF')
	plot(imf["temps"],imf["signal"],label='signal initial')
	plot(imf["temps"],imf["sommes_IMF"]+imf["residu"],'m',label='sommes des IMF')
	legend()

def init():
	affichage = 2
	itermax=2000 							# Parametres (nb cycles = nb. max d'IMF calculees et iterations = nombre d'iterations max pour 1 IMF)
	nb_cycles_max=5					# Critere de convergence pour l'obtention d'une enveloppe (si critere delta)
	delta = 1e-12							# Critere de convergence pour l'obtention d'une enveloppe (si critere SD)
	SD = 0.00003
	check_res = 1							# Pour chaque IMF, verification validite (0->non, 1->oui)
	debug_s=False						# Parametre de debuguage
	iter_ma = 6								#Nombre d'iteration du moving average
	facteur = 0.25							#Facteur de prolongement du signal
	crit_arret=1								#Choix du critere d'arret pour les iterations
	signal = 5									# choix du signal

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

	dx=temps[-1]/(len(temps)-1)
	nb_points=len(temps)
	return x,temps,affichage,itermax,nb_cycles_max,delta,SD,check_res,debug_s,iter_ma,facteur,dx,nb_points,crit_arret
	
	
x,temps,affichage,itermax,nb_cycles_max,delta,SD,check_res,debug_s,iter_ma,facteur,dx,nb_points,crit_arret=init()
sommes_IMF=np.zeros(len(x))
imf={}
imf["IMF"]={}
imf["IF"]={}
imf["signal"]=x
imf["temps"]=temps
test=test_non_oscil(x)
nb_IMF=1
signal_iter = x

while(test!=1) and (nb_IMF<=nb_cycles_max):
	print ' '
	print '=== Calcul de la IMF ',nb_IMF, '==='
	nb_iteration=0 
	s = signal_iter
	iter_s = True
	while (iter_s) and (nb_iteration<itermax):
		nb_iteration=nb_iteration+1
		#Recherche des extremas locaux
		min_,max_,t_extrema=cherche_extremas_locaux(s)
		# LMD 
		smoothed_mean,smoothed_magnitude = lmd(t_extrema,s,nb_points,iter_ma,facteur)
		# Calcul de la fonction h(t)
		h=s-smoothed_mean
		#Calcul du frequency modulated signal s(t)
		s_new=h/smoothed_magnitude
		# Calcul de l'enveloppe successive
		if nb_iteration==1:
			A=smoothed_magnitude
		else:
			A=np.multiply(A,smoothed_magnitude)
		#Test d'arret des iterations
		if crit_arret==1:
			sd=calcul_SD(s,s_new)
			if np.isnan(sd)==True:
				print 'probleme'
				iter_s=False
				nb_IMF=nb_cycles_max
				break
			if sd<SD:
				iter_s = False
			print 'iter',nb_iteration,' critere: ',sd
		elif crit_arret==2:
			# Abscisses des max et min de s
			min_s,max_s,inutile_ici = cherche_extremas_locaux(s_new)
			# Ordonnees des max et min de s
			min_s = s_new[min_s]
			max_s = s_new[max_s]
			try:
				if max(abs(max_s)-1)<delta: # on pourrait aussi prendre en compte le min
					iter_s = False
				print 'iter',nb_iteration,' critere: ',max(abs(max_s)-1)
			except:
				print 'iter',nb_iteration
				pass
		s=s_new

	#fichier = open('s.tex','w')
	#fichier.write('\\addplot[thick,color=blue] coordinates{\n')
	#for j in range (0,len(temps)):
	#	fichier.write('('+str("%.4f" % temps[j])+','+str("%.3f" % s[j])+')\n')
	#fichier.write('};\n')
	#Calcul de la i-eme IMF, ses phases et frequences instantannees
	[IMF,sommes_IMF,phi,omega]=calcul_IMF_IF(A,s,sommes_IMF,dx)
	# Creaction d'un dictionnaire contenant les IMF et les IF
	imf["IMF"]["IMF"+str(nb_IMF)]=IMF
	imf["IF"]["IF"+str(nb_IMF)]=omega
	nb_IMF = nb_IMF + 1
	signal_iter = signal_iter - IMF
	test=is_monotone(signal_iter)


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

imf["sommes_IMF"]=sommes_IMF
imf["nb_IMF"]=nb_IMF-1
imf["residu"]=signal_iter
visu(imf)
imf["opt_degen"]=True
imf["AMP"]={}


scipy.io.savemat("imf.mat",imf)
sauvegarde(imf,temps)
trace_pdf(temps,x,imf)

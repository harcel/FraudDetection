# Fraud detection models
# Call the functions with data, parameters and the hitlist. 
# The hitlist will be returned, extended with results of the model

import numpy as np
import numpy.random as rn
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pickle, os, time, sys, itertools, string

rn.seed(42)


######################################################################

def cost_per_member(data, params, hitlist, outdata, grouping='Prov_specialism'):
	"""
	This model compares the cost per member.

	There are three options for the grouping of providers:
	- Overall (not advised)
	- By known specialism, indicated by column with the grouping keyword, default='Prov_specialism'
	- By determined specialism. The model "billing_pattern" will be run in clustering mode
	
	For determining outliers there are also three options:
	- Above 90th percentile
	- Above 95th percentile
	- Statistical outlier (upper limit = Q3 + 1.5 (Q3-Q1))

	A version of the hitlist with the new results appended is returned

	"""

	option_group = params['Grouping']
	option_outlier = params['Outlier definition']

	if option_group == 'By determined peer group':
	    # Groups need to be determined
	    piv_proc = pd.pivot_table(data, values='Paid_amt', index='Provider_ID', columns='Procedure_code', aggfunc='sum')
	    piv_proc.replace(np.nan, 0, inplace=True)
	    fractional_proc = piv_proc.div(piv_proc.sum(axis=1), axis=0)
	    
	    from sklearn.decomposition import PCA
	    pca=PCA()
	    manifolds = pca.fit_transform(fractional_proc)[:,:3]
	    
	    from sklearn.cluster import DBSCAN
	    from sklearn.preprocessing import StandardScaler
	    X = StandardScaler().fit_transform(manifolds)
	    results = DBSCAN().fit(X)
	    
	    #This is one label per provider, in order, so I have to join these to the dataframe
	    tojoin = pd.DataFrame({'IDs':piv_proc.index, 'refgroup':results.labels_})
	    data = data.merge(tojoin, how='left', left_on='Provider_ID', right_on='IDs')
	    
	elif option_group == 'Overall': data['refgroup'] = np.zeros(len(data.index))
	elif option_group == 'Per specialism': data['refgroup'] = data.Prov_specialism
	else: 
	    print("The option for reference groups is not recognized! Overall is used.")
	    refgroup = np.zeros(len(data.index))
	    data['refgroup'] = refgroup

	refgroups = np.unique(data.refgroup)

	if option_outlier == 'Above 90th percentile': percentile = 90
	elif option_outlier == 'Above 95th percentile': percentile = 95
	elif option_outlier == 'Statistical outlier': pass  # Outside upper inner fence = q3+ 1.5*(q3-q1)
	else: print("Not yet implemented!")


	outliers = []
	score = []
	money = []

	# Definitions outdata
	outdata['costs'] = {'cost_per_member':[], 'Provider_ID':[], 'reference_group':[]}

	# plt.subplots_adjust(hspace=0.000)
	number_of_subplots=len(refgroups)

	maxval = 0
	minval = 1e9
	for iref, ref in enumerate(refgroups):
		if ref == -1 and option_group == 'By determined peer group': continue   # These are outliers determined by DBSCAN
		thisgroup = data[data.refgroup == ref]
		prov_pat = thisgroup.groupby(['Provider_ID', 'Member_ID'])
		cost_all_patients = pd.DataFrame(prov_pat.Paid_amt.sum())
		cost_all_patients.reset_index(inplace=True)
		per_prov = cost_all_patients.groupby('Provider_ID')
		cost_per_member = per_prov.Paid_amt.mean()
		number_patients = per_prov.Paid_amt.count()
		if cost_per_member.max() > maxval: maxval = cost_per_member.max()
		if cost_per_member.min() < minval: minval = cost_per_member.min()

		if option_group == 'Per specialism': plabel = str('Reference group ')+str(ref)
		elif option_group == 'By determined peer group': plabel = str('Reference group'+str(ref+1))
		elif option_group == 'Overall': plabel = ''
		else: plabel = ''

		if option_outlier in ['Above 90th percentile', 'Above 95th percentile']:
		    limval = np.percentile(cost_per_member, percentile)
		elif option_outlier == 'Statistical outlier':
		    q1, q3 = np.percentile(cost_per_member, [25, 75])
		    limval = q3 + 1.5 * (q3-q1)
		else: 
		    print("Outlier option not yet implemented!", option_outlier, "Using 90th percentile")
		    limval = np.percentile(cost_per_member, 90)

		median = np.median(cost_per_member)
		outdata['costs']['Provider_ID'].extend(list(cost_per_member.index))
		outdata['costs']['cost_per_member'].extend(list(cost_per_member.values))
		outdata['costs']['reference_group'].extend([ref]*len(cost_per_member))
		


		# ylims = ax.get_ylim()
		# ax.plot([limval, limval], ylims, color='red')
		# ax.plot([cost_per_member.max(), cost_per_member.max()], ylims, 'r:' )

		toomuch = cost_per_member[cost_per_member > limval]
		scoring_entities = toomuch.index
		outliers.extend(list(scoring_entities))

		score.extend(list((toomuch - limval) / np.abs(limval - median)))

		npats = number_patients[scoring_entities].values
		money.extend(list((toomuch-limval)*npats))

		# Add those numbers to the hitlist with the model name in it too.

	hl_add = pd.DataFrame({'Provider_ID':outliers, 'Score':score, 'Monetary':money, 
		'Model': ['Costs per patient']*len(score), 'Weight':[params['Weight']]*len(score) })

	hitlist = hitlist.append(hl_add, sort=True)

		
	return hitlist, outdata



######################################################################

def billing_pattern(data, params, hitlist, outdata):
	"""
	In this model, outliers from the general billing pattern (see below) are flagged,
	based on how far away from the nearest cluster they are.
	The pattern is defined as the fraction of money billed for different procedures.
	This multi-dimensional space is reduced to three dimensions. In that 3D space, a 
	DBSCAN cluster finder is used on the standardized locations, so the same density
	threshold for clusters can be used, no matter the values of input parameters.

	No monetary loss is defined, as it is not at all clear how this would be defined.
	"""

	# Parameters are passed for consistency, in this version it runs without any user-defined parameters.

	# Create a pivot table with amounts per procedure code for all providers, then normalize
	piv_proc = pd.pivot_table(data, values='Paid_amt', index='Provider_ID', columns='Procedure_code', aggfunc='sum')
	piv_proc.replace(np.nan, 0, inplace=True)
	fractional_proc = piv_proc.div(piv_proc.sum(axis=1), axis=0)
	

	# Create a lookup for the specialism
	prov_spec = data.loc[:,['Provider_ID', 'Prov_specialism']].drop_duplicates()
	prov_spec.set_index('Provider_ID', inplace=True)
	specs = np.array(prov_spec.values)

	# Use PCA to be able to select three main axes.
	from sklearn.decomposition import PCA
	pca=PCA()
	pcas = pca.fit_transform(fractional_proc)

	# Scale all axes to zero mean, unit stdev and do a density scan.
	from sklearn.preprocessing import StandardScaler
	from sklearn.cluster import DBSCAN
	X = StandardScaler().fit_transform(pcas[:,:3])
	scanner = DBSCAN(eps=0.5)
	results = scanner.fit(X)

	# Select outliers and compute scores
	# Compute, for all outliers, the distance to the nearest cluster center and normalize by stdev of that cluster.
	labels = results.labels_

	nclusters = len(np.unique(labels)) - (1 if -1 in labels else 0)

	# Calculate cluster centers and sizes
	center = np.zeros([nclusters, 4])

	for iclus in range(nclusters):
		coords = np.array(pcas[labels == iclus])
		center[iclus, :3] = np.array([np.mean(coords[:,0]), np.mean(coords[:,1]), np.mean(coords[:,2])])
		center[iclus, 3] = np.sqrt(np.std(coords[:,0])**2 + np.std(coords[:,1])**2 + np.std(coords[:,2])**2 )
	    

	out_pcas = pcas[labels == -1][:,:3]
	ids = piv_proc.index[labels == -1]

	outliers = list(ids)
	score = []
	money = []
	for pca in out_pcas:
		distsq = np.zeros(nclusters)
		for iclus in range(nclusters):
			distsq[iclus] = np.sum((np.array(pca) - np.array(center[iclus, :3]))**2)
		score.append(np.sqrt(np.min(distsq)) / (3*center[np.argmin(distsq), 3]))
		money.append(0)    

	# Definitions outdata
	outdata['billing'] = {'x':list(pcas[:,0]), 
						'y':list(pcas[:,1]), 
						'z':list(pcas[:,2]), 
						'Provider_ID':list(piv_proc.index), 
						'color':list(labels)}

	hl_add = pd.DataFrame({'Provider_ID':outliers, 'Score':score, 'Monetary':money, 
		'Model': ['Billing pattern']*len(score), 'Weight':[params['Weight']]*len(score)})

	hitlist = hitlist.append(hl_add, sort=True)



	return hitlist, outdata




######################################################################

def weekends_holidays(data, params, hitlist, outdata):
	"""
	Treatments on weekends and holidays are flagged. There is one parameter to rule this model:
	the choice to flag everybody who did so, or to flag people who do so significantly more often than others.

	Score is defined as treatments on the fraction of treatments on weekends and holidays,
	divided by the average of that over everybody who was flagged.
	Monetary loss is the sum of billed amounts on weekend days and holidays for those who are flagged. 
	In case only the ones with too many such treatments are flagged, it's multiplied by the fraction
	of treatments that was over the limit. 
	"""

	holidays = [dt.datetime(2016, 1, 1).date(), 
	            dt.datetime(2016, 12, 25).date(), 
	            dt.datetime(2016, 8, 10).date(), 
	            dt.datetime(2016, 11, 25).date()]


	data['holiday'] = [1 if (day.weekday() > 4 or day in holidays) else 0 for day in data.Treatment_date]
	data['holiday_money'] = data.holiday * data.Paid_amt

	per_prov = data.groupby('Provider_ID')
	n_treats = per_prov.holiday.count()
	n_holiday = per_prov.holiday.sum()
	money_holiday = per_prov.holiday_money.sum()
	frac_holiday = n_holiday / n_treats

	suspects = pd.concat([n_treats, n_holiday, money_holiday, frac_holiday], axis=1)
	suspects.columns=['n_treats', 'n_holiday', 'money_holiday', 'frac_holiday']

	suspects = suspects[suspects.n_holiday > 0]

	average_f = suspects.frac_holiday.mean()
	std_f = suspects.frac_holiday.std()

	if params['Flag'] == 'too many':
		suspects = suspects[suspects.frac_holiday > (average_f + 2*std_f)]
		corr_frac = suspects.frac_holiday - (average_f + 2*std_f)

	score = list(suspects.frac_holiday / suspects.frac_holiday.mean())
	money = list(suspects.money_holiday * (1 if params['Flag'] == 'all' else corr_frac))
	outliers = list(suspects.index)


	hl_add = pd.DataFrame({'Provider_ID':outliers, 'Score':score, 'Monetary':money, 
		'Model': ['Treatments on holidays and weekends']*len(score), 'Weight':[params['Weight']]*len(score)})

	hitlist = hitlist.append(hl_add, sort=True)


	return hitlist, outdata




######################################################################

def rising_revenue(data, params, hitlist, outdata):
	"""
	This detection model checks if there are any signs of significantly rising activity/revenue, 
	throughout the period of activity. Both gradually rising, as well as step functions in activity
	will be detected through the comparison of regression models of cumulative revenue over time.

	With steady activity, cumulative revenue should be linear with time, with the revenue per unit time 
	as slope. With a steadily rising revenue, the cure will be more parabola like, and a linear regression
	is likely to result in too low an average revenue per unit time. A step function in activity results 
	in a different slope before and after the step and in case of a rising revenue, this will again look
	a little parabola-like and result in too low an averge revenue per unit time if estimated with linear
	regression. Both will also result in a low goodness-of-fit for a linear relation. 

	As a good proxy, the average slope of the cumulative revenue is compared to the total one.
	The distribution of the ratios of the two is roughly normal. Everything above 2 sigma deviation
	is flagged, and the score is equal to the dev in stds - maximum allowed deviation (2std).
	"""

	# Revenue per month, per provider

	data['month'] = [d.month for d in data.Treatment_date]
	perprovpermonth = data.groupby(['Provider_ID', 'month'])
	revpermonth = perprovpermonth.Paid_amt.sum()


	cumrev = revpermonth.groupby(level=[0]).cumsum().reset_index()
	cumrev['slopes'] = cumrev.Paid_amt / cumrev.month

	av_slope = cumrev.groupby('Provider_ID').slopes.mean()

	rpm = revpermonth.reset_index()
	perprov = rpm.groupby('Provider_ID')
	tot_money = perprov.Paid_amt.sum() / 12

	stds = ((tot_money / av_slope)-1) /  np.std(tot_money / av_slope) 

	maxstd = 2
	stds = stds[stds.values > maxstd] - maxstd

	# Definition outdata
	outdata['increasing_revenue'] = {}
	outdata['increasing_revenue']['Provider_ID'] = list(stds.index)
	outdata['increasing_revenue']['Values'] = []
	outdata['increasing_revenue']['Months'] = []
	for prov in stds.index: 
		outdata['increasing_revenue']['Months'].append(list(revpermonth[prov].index))
		outdata['increasing_revenue']['Values'].append(list(revpermonth[prov].values))

	hl_add = pd.DataFrame({'Provider_ID':stds.index, 'Score':stds.values, 'Monetary':[0]*len(stds), 
		'Model': ['Rising revenue']*len(stds), 'Weight':[params['Weight']]*len(stds)})

	hitlist = hitlist.append(hl_add, sort=True)


	return hitlist, outdata


######################################################################

def seasonality(data, params, hitlist, outdata):
	"""
	In summer, patients go on holidays, so there is a trend that fewer patients are treated in July and August,
	which is (partly) made up by them showing up in September and/or October. This trend is visible in the 
	overall data, with all ehalythcare providers included. 
	This model checks if every provider with enough volume shows this behavior too (so a larger deviation 
	is allowed for those with smaller volume, to correct for Poisson noise in the treatment dates). 

	"""


	# Revenue per 2 months, per provider

	data['qmonth'] = [np.floor((d.month - 1)/2) for d in data.Treatment_date]
	perprovpermonth = data.groupby(['Provider_ID', 'qmonth'])
	revpermonth = perprovpermonth.Paid_amt.sum().reset_index()
	uncertainty = (np.sqrt(perprovpermonth.Paid_amt.count()) / (perprovpermonth.Paid_amt.count())).reset_index()
	uncertainty.rename(columns={'Paid_amt':'uncertainty'}, inplace=True)

	#Normalize over everyone to find total fracs in all qmonths
	fracs = (data.groupby('qmonth').Paid_amt.sum() / np.sum(data.Paid_amt)).reset_index()

	# Same thing per provider, including a measure of uncertainty, based on volume
	revperprov = data.groupby('Provider_ID').Paid_amt.sum().reset_index()

	revs = pd.merge(revpermonth, revperprov, left_on='Provider_ID', right_on='Provider_ID', how='left')
	revs['fracprov'] = revs.Paid_amt_x / revs.Paid_amt_y

	withun = revs.merge(uncertainty, left_on=['Provider_ID', 'qmonth'], right_on=['Provider_ID', 'qmonth'], how='left')
	allnumbers = withun.merge(fracs, left_on='qmonth', right_on='qmonth', how='left')
	allnumbers.rename(columns={'Paid_amt':'totfrac'}, inplace=True)

	allnumbers['deviation'] = ((allnumbers.fracprov - allnumbers.totfrac) / allnumbers.totfrac) / allnumbers.uncertainty
	scores = abs(allnumbers.groupby('Provider_ID').deviation.mean()) - 1
	scores = scores[scores.values > 0]
		
	outdata['seasonality'] = {}
	outdata['seasonality']['Provider_ID'] = list(scores.index)
	outdata['seasonality']['Values'] = []
	outdata['seasonality']['Months'] = []
	for prov in scores.index: 
		outdata['seasonality']['Months'].append(list(allnumbers[allnumbers.Provider_ID == prov].qmonth))
		outdata['seasonality']['Values'].append(list(100*allnumbers[allnumbers.Provider_ID == prov].fracprov))
	outdata['seasonality']['Month_names'] = ['Jan-Feb','Mar-Apr','May-Jun','Jul-Aug','Sep-Oct','Nov-Dec']
	outdata['seasonality']['Entire_population'] = 100*fracs.Paid_amt

	hl_add = pd.DataFrame({'Provider_ID':scores.index, 'Score':scores.values, 'Monetary':[0]*len(scores), 
		'Model': ['Seasonality']*len(scores), 'Weight':[params['Weight']]*len(scores)})

	hitlist = hitlist.append(hl_add, sort=True)


	return hitlist, outdata


######################################################################

def combination_codes(data, params, hitlist, outdata):
	""" Codes I1 and H1 should not appear together 
	"""

	combinlines = pd.concat( [data[(data.Procedure_code == "I1")], data[(data.Procedure_code == "H1")]])
	combinlines.sort_values(["Provider_ID", "Member_ID", "Treatment_date"], inplace=True)

	per_visit = combinlines.groupby(["Provider_ID", "Member_ID", "Treatment_date"])
	nlines = per_visit.Procedure_code.nunique()

	nl = pd.DataFrame(nlines).reset_index()

	per_prov = nl.groupby('Provider_ID')
	ncombis = per_prov.Procedure_code.sum() - per_prov.Procedure_code.count()
	ncombis = ncombis[ncombis.values > 0] 

	# Scores such that median = 1, monetary loss is H1 prize per doucle line.
	mednumber = ncombis.median()
	ids = ncombis.index
	score = [p/mednumber for p in ncombis.values]
	money = [34 * p for p in ncombis.values]

	# Make the monetary loss equal to the rate for H1: 34.

	hl_add = pd.DataFrame({'Provider_ID':ids, 'Score':score, 'Monetary':money, 
		'Model': ['Combination codes']*len(score), 'Weight':[params['Weight']]*len(score)})

	hitlist = hitlist.append(hl_add, sort=True)


	return hitlist, outdata



######################################################################

def fraction_expensive(data, params, hitlist, outdata):
	""" An outlier detection is performed on the fraction of expensive versus cheap
	versions of the same procedure (encoded as A1 vs A2, for example).
	User specified options are:
	- determined per specialism or overall 
	- Statistical outliers in the sample, or every provider above 90th percentile is flagged
	- Outliers are determined in number of cheap vs expensive treatments, or in revenue in 
		cheap vs expensive versions.

	Scores depend on severity of outlier, monetary losses are undefined if outliers are 
		sought in number of treatments and are defined as the fraction of the expensive revenue
		that is outside the accepted range.
	"""


	# Parameter in settings, per specialism or overall
	grouping = params['Grouping'] #'per_specialism' or 'overall'
	if   grouping == 'overall':        data['Specialism'] = [1 for d in data.Prov_specialism]
	elif grouping == 'per_specialism': data['Specialism'] = [d for d in data.Prov_specialism]

	# Parameter in settings: ratio in numbers or in price
	ratio = params['Ratio']   # options: 'number' or 'price'

	# Parameter in settings: statistical outlier or above 90th percentile
	outlier = params['Outlier']   # options: 'statistical' or 'percentile'

	data['procedure_group'] = [d[0] for d in data.Procedure_code]
	data['price_group'] = ['Expensive' if float(d[1]) > 1 else 'Cheap' for d in data.Procedure_code]

	pergroup = data.groupby(['Specialism', 'Provider_ID', 'procedure_group', 'price_group'])
	if ratio == 'number': 
	    quantity = pergroup.Paid_amt.count()
	elif ratio == 'price':
	    quantity = pergroup.Paid_amt.sum()

	qs = pd.DataFrame(quantity).reset_index()

	pivot = pd.pivot_table(qs, index=['Specialism', 'Provider_ID', 'procedure_group'], columns=['price_group'], 
	                       values='Paid_amt').reset_index()
	perprov = pivot.groupby(['Specialism', 'Provider_ID'])

	totcheap = pd.DataFrame(perprov.Cheap.sum())
	totexp = pd.DataFrame(perprov.Expensive.sum())

	totals = pd.merge(totcheap, totexp, left_index=True, right_index=True).reset_index()
	totals['frac_exp'] = [(d[1]/(d[0]+d[1])) for d in zip(totals.Cheap, totals.Expensive)]

	for spec in np.unique(totals.Specialism):
		thisspec = totals[((totals.Specialism == spec) & (totals.frac_exp > -1))]
		if outlier == 'statistical':
			q1, q3 = np.percentile(thisspec.frac_exp, [25, 75])
			limval = q3 + 1.5 * (q3-q1)
		elif outlier == 'percentile':
			limval = np.percentile(thisspec.frac_exp, 90)
		else: print("Unknown outlier definition")
		outliers = thisspec[thisspec.frac_exp > limval]
		outliers['score'] = (outliers.frac_exp - limval) / np.median(outliers.frac_exp - limval)
		if   ratio == 'number': outliers['money'] = np.zeros(len(outliers.score))
		elif ratio == 'price': outliers['money'] = [(d[0]-limval)*d[1] for d in zip(outliers.frac_exp, outliers.Expensive)]


		hl_add = pd.DataFrame({'Provider_ID':outliers.Provider_ID, 'Score':outliers.score, 'Monetary':outliers.money, 
		        'Model': ['Fraction expensive treatments']*len(outliers.score), 
		                       'Weight':[params['Weight']]*len(outliers.score)})

		hitlist = hitlist.append(hl_add, sort=True)

	return hitlist, outdata


######################################################################


def freely_billed(data, params, hitlist, outdata):
	""" An outlier detection is performed on billed rate per freely billable procedure.
	User specified options are:
	- determined per specialism or overall 
	- Statistical outliers in the sample, or every provider above 90th percentile is flagged
	- Outliers are determined over all frely billable procedures combined, or per 
		procedure group (indicated by first letter of procedure code)

	Scores depend on severity of outlier, monetary losses are defined as the difference in revenue 
		between the real value, and what it would be if all freely billed procedures had on average
		the value that is determined to be the maximum acceptable, from the data.
	"""


	grouping = params['Grouping']  # options: 'overall' or 'per_specialism'
	if   grouping == 'overall':        data['Specialism'] = [1 for d in data.Prov_specialism]
	elif grouping == 'per_specialism': data['Specialism'] = [d for d in data.Prov_specialism]

	group_codes = params['Group_codes'] # options: 'per_procedure', 'overall'
	 
	outlier = params['Outlier'] # options: 'statistical', 'percentile'
	    
	# Procedures with free rates: C, F, I
	data['proc_group'] = [d[0] for d in data.Procedure_code]
	freerates = data[data.proc_group.isin(['C', 'F', 'I'] ) ]

	if group_codes == 'overall': freerates['proc_group'] = ['X'] * len(freerates.proc_group)


	for thisgroup in np.unique(freerates.proc_group):
		for thisspec in np.unique(freerates.Prov_specialism):
			subset = freerates[((freerates.proc_group == thisgroup) & (freerates.Prov_specialism == thisspec))]

			perprov = subset.groupby('Provider_ID')
			meanrate = pd.DataFrame(perprov.Paid_amt.mean())
			totalmoney = pd.DataFrame(perprov.Paid_amt.sum())

			results = pd.merge(meanrate, totalmoney, left_index=True, right_index=True)

			if outlier == 'statistical':
				q1, q3 = np.percentile(results.Paid_amt_x, [25, 75])
				limval = q3 + 1.5 * (q3-q1)
			elif outlier == 'percentile':
				limval = np.percentile(results.Paid_amt_x, 90)
			else: print("Unknown outlier definition")
			    
			outliers = results[results.Paid_amt_x > limval]
			outliers['score'] = (outliers.Paid_amt_x - limval) / np.median(outliers.Paid_amt_x - limval)
			outliers['money'] = [d[1]-(d[1]/d[0]*limval) for d in zip(outliers.Paid_amt_x, outliers.Paid_amt_y)]
			# print(outliers)

			hl_add = pd.DataFrame({'Provider_ID':outliers.index, 'Score':outliers.score, 'Monetary':outliers.money, 
			    'Model': ['Rate per freely billed procedure']*len(outliers.score), 
			                   'Weight':[params['Weight']]*len(outliers.score)})

			hitlist = hitlist.append(hl_add, sort=True)

	return hitlist, outdata


######################################################################


def periodic_often(data, params, hitlist, outdata):
	""" Periodic treatments (code BX) should be done once or twice a year. 
	Catch providers who have done more than that to some patients. 
	"""

	maxnumber = params['maxnumber']

	data['p_group'] = [b[0] for b in data.Procedure_code]
	data_B = data[data.p_group == 'B']

	per_patient = data_B.groupby(['Provider_ID', 'Member_ID'])
	n_t = per_patient.Treatment_date.nunique()

	hits = pd.DataFrame(n_t[n_t.values > maxnumber] - maxnumber)
	hits.reset_index(inplace=True)
	per_prov = hits.groupby('Provider_ID')
	nh = per_prov.Treatment_date.sum()
	med = nh.median()

	scores = list(nh / med)
	ids = list(nh.index)	
	money = nh.values * 23 # 23 is the rate for B1

	hl_add = pd.DataFrame({'Provider_ID':ids, 'Score':scores, 'Monetary':money, 
		'Model': ['Periodic treatment too often']*len(scores), 
       	'Weight':[params['Weight']]*len(scores)})

	hitlist = hitlist.append(hl_add, sort=True)

	return hitlist, outdata


if __name__ == "__main__":
	print("No tests implemented yet.")

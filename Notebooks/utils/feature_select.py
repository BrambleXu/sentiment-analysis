import heapq 


def chi(term, cl, X, Y):
	N = len(Y)
	A = B = C = D = 0
	for i, x in enumerate(X):
		flag = False
		for k in x.keys():
			if k == term:
				flag = True
				break
		if flag == True:
			if str(Y[i]) == str(cl):
				A = A + 1
			else:
				B = B + 1
		else:
			if str(Y[i]) == str(cl):
				C = C + 1
			else:
				D = D + 1
	re = float((N * (A * D - B * C) * (A * D - B * C)) / ((A + B) * (A + C) * (B + D) * (C + D)))
	return re


# def select_feature(X, Y, dic, cl, k):
# 	heap = []
# 	heapq.heapify(heap)
# 	
# 	for t in dic.keys():
# 		val = chi(t, cl, X, Y)
# 		#print t + '\t' + str(val)
# 		if len(heap) < k:
# 			heapq.heappush(heap, (val, t))
# 		else:
# 			if val > heap[0][0]:
# 				heapq.heappop(heap)
# 				heapq.heappush(heap, (val, t))
# 
# 	return [heapq.heappop(heap)[1] for i in range(len(heap))]


def select_feature(filePath, k):
	read = open(filePath, 'r')
	lab_fea = {}
	
	for line in read:
		line_arr = line.strip().split()
		if len(line_arr) - 1 <= k:
			lab_fea[line_arr[0]] = [kv.split(':')[0] for kv in line_arr[1 : ]]
		else:
			heap = []
			heapq.heapify(heap)
			for kv in line_arr[1 : ]:
				key, val = kv.split(':')
				if len(heap) < k:
					heapq.heappush(heap, (float(val), key))
				else:
					if float(val) > heap[0][0]:
						heapq.heappop(heap)
						heapq.heappush(heap, (float(val), key))
			lab_fea[line_arr[0]] = [heapq.heappop(heap)[1] for i in range(len(heap))]
	read.close()
	return lab_fea
	
	
	
def dump_feature(lab, dic, X, Y, filePath):
	out = open(filePath, 'w')
	dic_num = len(dic)
	for cl in lab:
		out.write(str(cl))
		i = 0
		for term in dic.keys():
			if( (i + 1) % 1000 == 0 ):
				print("Term %d of %d" % (i + 1, dic_num))
			s = chi(term, cl, X, Y)
			out.write(' ' + term + ':' + str(s))
			i += 1
		out.write('\n')
		out.flush()
	out.close()
	
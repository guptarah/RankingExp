#! /usr/bin/python

"""
This moudle implements various correlation measures such as,
Spearman rank correlation coefficients,
Kendall rank correlation coefficients,
Average Continuity, and
Pearson product moment coefficient.
It provides confidence intervals where applicable.

----------------REVISIONS-----------------------------
2009-07-08: Modified Spearman computations to handle equal ranks.

Danushka Bollegala

Copyright (c) 2009, Danushka Bollegala
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list
     of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation and/or
      other materials provided with the distribution.
    * Neither the name of the <ORGANIZATION> nor the names of its contributors
     may be used to endorse or promote products derived from this software 
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys,math,random

class CORRELATION:

    def __init__(self):
        """
        for non-numeric objects, we map to a unique numeric ID.
        """
        self.rankMap = {}
        pass

    def clearRankMap(self):
        """
        resets the unique numeric ID map.
        """
        self.rankMap = {}
        pass

    def comparator(self,p,q):
        # prefer elements with lower scores.
        if p[0] > q[0]:
            return 1
        elif p[0] == q[0]:
            # if scores are equal, prefer elements with lower ranks.
            if p[1] > q[1]:
                return 1
        return -1

    def RandomShuffle(self,A,B):
        """
        Randomly shuffle the indexes (not elements) in the two lists A, B.
        Note that for the same index, we must have the original values that
        were in A and B for that index even after shuffling. This must be
        done because when computing Spearman, if two or more word pairs
        are assigned with the same score, then they would not get sorted.
        Instead they will remain in the order they appear in the benchmark
        dataset. This will give an unfair boost to methods that assign the
        same similarity score (i.e. 0 or 1) to many word pairs in the dataset.
        However, this approach is problematic because the computed Spearman
        score will change randomly. Instead of doing RandomShuffling it is
        better to do a deterministic ranking considering equal valued items
        in a list, as done in ConvertToRanks function.
        """
        if len(A) != len(B):
            raise "Correlation.py:RandomShuffle. No of items are different!"
        N = len(A)
        L = range(0,N)
        L.sort(self.__randComp__)
        sA = A[:]
        sB = B[:]
        c = 0
        for i in L:
            sA[c] = A[i]
            sB[c] = B[i]
            c += 1
        return (sA,sB)

    def __randComp__(self,A,B):
        """
        Generate a random number and return a sorting decision based
        upon the generated random number.
        """
        randVal = random.random()
        if randVal > 0.5:
            return 1
        else:
            return -1
        pass

    def ConvertToRanks_equals(self,scoresList):
        """
        Assign ranks (from 0) to elements in scoresList.
        top ranks are assigned to low scores.
        Equal ranks are assigned to elements with equal scores.
        The rank assigned to equal scored elements is the
        aveage of their ranks.
        """
        # count how many elements have each score.
        L = []
        h = {}
        scoreToRank = {}
        for i in range(0,len(scoresList)):
            score = scoresList[i]
            h[score] = h.get(score,0)+1
            L.append((score,float(i)))
        # sorts in the ascending order of the score.
        L.sort(self.comparator)
        # modify the rank of equal scored elements.
        ranks = []
        for (score,r) in L:
            if h[score] == 1:
                ranks.append(r)
                scoreToRank[score] = r
            else:
                if score not in scoreToRank:
                    avg = r+(1.0/float(h[score]))
                    scoreToRank[score] = avg
                    ranks.append(avg)
                else:
                    ranks.append(scoreToRank[score])                
        return ranks

    def ConvertToRanks(self,scoresList):
        #return self.ConvertToRanks_no_equals(scoresList)
        return self.ConvertToRanks_equals(scoresList)  
    
    def ConvertToRanks_no_equals(self,scoresList):
        """
        Assign ranks (from 0) to elements in scoresList.
        top ranks are assigned to low scores.
        """
        L = []
        for i in range(0,len(scoresList)):
            L.append((scoresList[i],i))
        L.sort(self.comparator)
        ranks = []
        for (score,r) in L:
            ranks.append(r)
        return ranks

    def AssignIDs(self,valueList):
        """
        Assign a a unique numeric ID to values in valueList.
        """
        r = 0
        rankMap = {}
        ranks = []
        for val in valueList:
            if val not in self.rankMap:
                self.rankMap[val] = r
                ranks.append(r)
                r += 1
            else:
                ranks.append(self.rankMap[val])
        return ranks

    def AlignLists(self,A,refLists):
        """
        Changes values in refLists s.t. they corresponds to ranks of A.
        """
        L = []
        for refL in refLists:
            sL = []
            for e in refL:
                sL.append(A.index(e))
            L.append(sL)
        return L

    def Spearman(self,A,B):
        """
        Compute Spearman correlation between two ranked lists.
        Convert scores to ranks and then use Pearson.
        """
        if len(A) != len(B):
            raise "The two lists have different lengths!"
        rA = self.ConvertToRanks_equals(A)
        rB = self.ConvertToRanks_equals(B)
        print rA, rB
        return self.Pearson(rA,rB)

    def MultipleSpearman(self,A,L):
        """
        Computes the extended Spearman correlation coefficient by considering
        the list under evaluation (A), against multiple reference lists, (L).
        e.g. L = [[1,2,3],[1,3,2]]
        """
        rankLists = []
        rA = self.ConvertToRanks(A)
        for scoreList in L:
            if len(A) != len(scoreList):
                raise "The two lists have different lengths!"
            rankLists.append(self.ConvertToRanks(scoreList))
        refList = self.AlignLists(rA,rankLists)
        return self.__spearman__(refList)

    def __spearman__multi__(self,refLists):
        """
        Actual computation of Spearman goes here.
        calculates spearman rank correlation coefficient.
        refLists is a list of integer sequence lists indicating the position
        of sentences with respect to reference summary.
        Eg. two reference summaries being assigned IDs with respect to
        a evaluation summary. [[1,2,3], [2,1,3]] 
        """
        D = 0
        N = len(refLists[0])
        for i in xrange(0,N):
            #we will consider only the minimum distance!
            min_dist = float("infinity")
            for refList in refLists:
                dist = (i-refList[i])**2
                if dist < min_dist:
                    min_dist = dist
            D = D+min_dist
        sp=float(D)/float((N*N*N)-N)
        sp=1-(6.0*sp)
        return sp

    def __kendall__(self,refLists):
        """
        refLists is a list of integer sequence lists indicating the position
        of sentences with respect to reference summary.
        Eg. two reference summaries being assigned IDs with respect to
        a evaluation summary. [[1,2,3], [2,1,3]] 
        """
        nd = 0 ##discordant pairs
        N = len(refLists[0])
        for i in xrange(0,N):
            for j in xrange(i+1,N):
                #if we find a concordant pair in at least one of
                #the reference summaries, then that is enough.
                found = False
                for refList in refLists:
                    if refList[j]>refList[i]:
                        found = True
                        break
                    pass
                if not found:
                    nd += 1
                    pass
                pass
            pass
        tau=float(4*nd)/float(N*(N-1))
        tau=1-tau
        return tau

    def MultipleKendall(self,A,L):
        """
        Computes the extended Kendall correlation coefficient by considering
        the list under evaluation (A), against multiple reference lists, (L).
        e.g. L = [[1,2,3],[1,3,2]]
        """
        rankLists = []
        rA = self.ConvertToRanks(A)
        for scoreList in L:
            if len(A) != len(scoreList):
                raise "The two lists have different lengths!"
            rankLists.append(self.ConvertToRanks(scoreList))
        refList = self.AlignLists(rA,rankLists)
        return self.__kendall__(refList)
        pass

    def MultipleAverageContinuity(self,A,L):
        """
        Computes the extended average continuity measure by considering
        the list under evaluation (A), against multiple reference lists, (L).
        e.g. L = [[1,2,3],[1,3,2]]
        """
        rankLists = []
        rA = self.ConvertToRanks(A)
        for scoreList in L:
            if len(A) != len(scoreList):
                raise "The two lists have different lengths!"
            rankLists.append(self.ConvertToRanks(scoreList))
        refList = self.AlignLists(rA,rankLists)
        return self.__AverageContinuity__(refList)
        pass

    def Pearson(self,A,B):
        """
        Compute Pearson coefficient between two numeric lists.
        """
        N=len(A)
        if N!=len(B):
            sys.stderr.write("Cannot compute pearson. Lengths mismatch\n")
            sys.exit(1)
            pass
        totA = sum(A)
        totB = sum(B)
        meanA = totA/float(N)
        meanB = totB/float(N)
        sA = 0
        sB = 0
        for i in range(0,N):
            sA = sA+((A[i]-meanA)*(A[i]-meanA))
            sB = sB+((B[i]-meanB)*(B[i]-meanB))
            pass
        sA = sA/float(N)
        sB = sB/float(N)
        sA = math.sqrt(sA)
        sB = math.sqrt(sB)    
        pSum = 0
        for i in range(0,N):
            pSum = pSum+((A[i]-meanA)*(B[i]-meanB))
            pass
        if sA*sB == 0:
            return 0
        else:
            P = pSum/(float(N)*sA*sB)
            return P
        pass

    def Kendall(self,A,B):
        """
        Compute Kendall tau between two ranked lists.
        """
        if len(A) != len(B):
            raise "The two lists have different lengths!"
        rA = self.ConvertToRanks(A)
        rB = self.ConvertToRanks(B)
        refList = self.AlignLists(rA,[rB])
        return self.__kendall__(refList)
        pass

    def AverageContinuity(self,A,B):
        """
        Compute average continuity metric between two ranked lists.
        """
        if len(A) != len(B):
            raise "The two lists have different lengths!"
        rA = self.ConvertToRanks(A)
        rB = self.ConvertToRanks(B)
        refList = self.AlignLists(rA,[rB])
        return self.__AverageContinuity__(refList)
        pass

    def __AverageContinuity__(self,refLists):
        """
        calculate average continuity
        """
        UpTo=5 #Upto and not including lenghts for n-grams
        #if this was 5, then consider 2,3,4 grams!
        sumLen=len(refLists[0])
        Ngrams=[]
        #add zeros to Ngrams
        alpha=0.01
        for i in range(0,max(9,sumLen+1)):
            Ngrams.append(0)    
        for i in range(0,sumLen):
            for j in range(i+1,sumLen):
                seq = range(i,j+1)
                votes = 0
                for refList in refLists:
                    votes = votes+IsSubList(refList,seq)
                if votes>0:
                    Ngrams[len(seq)] = Ngrams[len(seq)]+1
        resList = [] #holds results, first element is AC, rest is precision
        resList.append(0)    
        for i in range(2,9):
            resList.append(float(Ngrams[i]))
        #calculating average continuity
        avgCont = float(0)
        for i in range(2,UpTo):
            norm = (float(Ngrams[i])+alpha)/(float(sumLen-i+1)+alpha)
        avgCont = avgCont+math.log(norm)
        avgCont = avgCont/float(UpTo-2)    
        avgCont = math.exp(avgCont)
        resList[0] = avgCont    
        for i in range(1,len(resList)):
            if resList[i] !=0:
                resList[i] = resList[i]/float(sumLen-(i+1)+1)    
        return(resList)

    def PearsonConfidence(self,r,n):
        """
        Returns the confidence interval (95%) for the Pearson correlation
        coefficient r and sample size n.
        """
        if r >= 1 or r <= -1:
            return (r,r)
        if n < 4:
            raise "Must have at least four points to compute confidence intervals"
        #sys.stderr.write("%f\n" % r)
        # compute Fisher transformation F(r)=0.5*(log(1+r)/(1-r))
        fr = 0.5*math.log((1.0+r)/(1-r))
        # compute Z score. z=fr*sqrt(n-3)
        z = fr*math.sqrt(float(n)-3)
        # 95% confidence values at 1.96
        critical = 1.96
        zlow = z-critical
        zup = z+critical
        # convert back to Fisher
        flow = zlow/math.sqrt(float(n)-3)
        fup = zup/math.sqrt(float(n)-3)
        rup = (math.exp(2*fup)-1)/(math.exp(2*fup)+1)
        rlow = (math.exp(2*flow)-1)/(math.exp(2*flow)+1)
        return (rlow,rup)

    def Normalize(self,L):
        """
        Normalize the values in L to [0,1] scale.
        The list L itself is changed. Keep a copy before you call this
        function if you want the original list. The function returns
        the mean and standard deviation of the list.
        """
        sqdTot = float(sum([x*x for x in L]))
        tot = float(sum(L))
        N = float(len(L))
        myu = tot/N
        sigma = (sqdTot/N)-(myu*myu)
        sigma = math.sqrt(sigma)
        for i in range(0,len(L)):
            L[i] = (L[i]-myu)/sigma
        minL = min(L)
        for i in range(0,len(L)):
            L[i] = L[i]-minL
        maxL = max(L)
        for i in range(0,len(L)):
            L[i] = L[i]/maxL
        return {"mean":myu,"SD":sigma}

    def t_test(self,A,B):
        """
        Perform a Student's t-test, for dependent paired two-tailed case.
        """
        from statlib import stats
        (tval,p) = stats.ttest_rel(A,B,1)
        return {"tval":tval,"prob":p}
    pass


def main():
    C = CORRELATION()
    A = [1,2,3,4,5]
    B = [2,3,4,4,9]
    r = C.Spearman(A,B)
    n = len(A)
    print A, B,
    print "Pearson Correlation =", C.Pearson(A,B)
    print "Spearman Correlation =", C.Spearman(A,B)
    pass

if __name__ == "__main__":
    main()

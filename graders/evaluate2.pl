#!/usr/bin/perl -s
use utf8;
binmode(STDOUT, ":utf8");
binmode(STDIN, ":utf8");

# gold standard results all
my @goldstandardresults = ("0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "2", "2", "0", "0", "2", "0", "0", "1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "2", "2", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "2", "2", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "2", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "2", "0", "0", "0", "0", "2", "0", "0", "2", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "1", "0", "0", "2", "0", "0", "0", "0", "0", "0", "2", "2", "2", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "1", "0", "0", "0", "0", "2", "0", "0", "2", "2", "0", "0", "0", "1", "0", "0", "0", "2", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "2", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "2", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "2", "2", "2", "0", "0", "0", "0", "2", "0", "0", "0", "0", "2", "0", "0", "2", "0", "0", "2", "2", "0", "0", "0", "0", "2", "0", "0", "2", "0", "2", "1", "0", "0", "0", "0", "2", "0", "2", "0", "0", "0", "0", "0", "0", "1", "0", "2", "0", "0", "0", "2", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "2", "2", "0", "0", "0", "2", "0", "0", "2", "0", "1", "0", "0", "2", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "2", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "2", "0", "0", "0", "0", "2", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "1", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "2", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "2", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "2", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "2", "0", "2", "0", "0", "0", "0", "2", "0", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "2", "2", "0", "0", "0", "0", "0", "2", "2", "2", "0", "0", "0", "0", "0", "0", "0", "2", "0", "0", "0", "0", "2", "0");

my $participantresults = $ARGV[0];

# 0=866
# 1=25
# 2=109

open(FILE, "$participantresults") or die "Cannot open!";
my @participantresults_all = <FILE>;
close FILE;
chomp(@participantresults_all);

my $TP0=0;
my $TP1=0;
my $TP1=0;
my $TN0=0;
my $TN1=0;
my $TN2=0;
my $FN0=0;
my $FN1=0;
my $FN2=0;
my $FP0=0;
my $FP1=0;
my $FP2=0;

foreach my $index (0..999){
	if ($goldstandardresults[$index]==0 && $participantresults_all[$index]==0){
		$TP0++;
	} 
	if ($goldstandardresults[$index]!=0 && $participantresults_all[$index]==0) {
		$FP0++;
	} 
	if ($goldstandardresults[$index]==0 && $participantresults_all[$index]!=0) {
		$FN0++;
	} 
	if ($goldstandardresults[$index]!=0 && $participantresults_all[$index]!=0) {
		$TN0++;
	} 
	if ($goldstandardresults[$index]==1 && $participantresults_all[$index]==1){
		$TP1++;
	} 
	if ($goldstandardresults[$index]!=1 && $participantresults_all[$index]==1) {
		$FP1++;
	} 
	if ($goldstandardresults[$index]==1 && $participantresults_all[$index]!=1) {
		$FN1++;
	} 
	if ($goldstandardresults[$index]!=1 && $participantresults_all[$index]!=1) {
		$TN1++;
	} 
	if ($goldstandardresults[$index]==2 && $participantresults_all[$index]==2){
		$TP2++;
	} 
	if ($goldstandardresults[$index]!=2 && $participantresults_all[$index]==2) {
		$FP2++;
	} 
	if ($goldstandardresults[$index]==2 && $participantresults_all[$index]!=2) {
		$FN2++;
	} 
	if ($goldstandardresults[$index]!=2 && $participantresults_all[$index]!=2) {
		$TN2++;
	}

}


# micro averages
my $microAveragePrecision = ($TP0+$TP1+$TP2)/($TP0+$TP1+$TP2+$FP0+$FP1+$FP2+0.0000000000000000000001);
my $microAverageRecall = ($TP0+$TP1+$TP2)/($TP0+$TP1+$TP2+$FN0+$FN1+$FN2+0.0000000000000000000001);
my $microAverageFscore = 2*$microAveragePrecision*$microAverageRecall/($microAveragePrecision+$microAverageRecall+0.0000000000000000000001);

my $accuracy0=($TP0+$TN0)/($TN0+$TP0+$FP0+$FN0+0.0000000000000000000001);
my $precision0=$TP0/($TP0+$FP0+0.0000000000000000000001);
my $recall0=$TP0/($TP0+$FN0+0.0000000000000000000001);
# my $balancedf0=2*$precision0*$recall0/($precision0+$recall0+0.0000000000000000000001);

my $accuracy1=($TP1+$TN1)/($TN1+$TP1+$FP1+$FN1+0.0000000000000000000001);
my $precision1=$TP1/($TP1+$FP1+0.0000000000000000000001);
my $recall1=$TP1/($TP1+$FN1+0.0000000000000000000001);
# my $balancedf1=2*$precision1*$recall1/($precision1+$recall1+0.0000000000000000000001);

my $accuracy2=($TP2+$TN2)/($TN2+$TP2+$FP2+$FN2+0.0000000000000000000001);
my $precision2=$TP2/($TP2+$FP2+0.0000000000000000000001);
my $recall2=$TP2/($TP2+$FN2+0.0000000000000000000001);
# my $balancedf2=2*$precision2*$recall2/($precision2+$recall2+0.0000000000000000000001);

my $macroAveragePrecision = ($precision0+$precision1+$precision2)/3;
my $macroAverageRecall = ($recall0+$recall1+$recall2)/3;
my $macroAverageFscore = 2*$macroAveragePrecision*$macroAverageRecall/($macroAveragePrecision+$macroAverageRecall+0.0000000000000000000001);

$microAverageFscore = sprintf("%.2f", $microAverageFscore*100);
$macroAverageFscore = sprintf("%.2f", $macroAverageFscore*100);

print "Micro-Average F-score = $microAverageFscore\%\n";
# print "Macro-Average Precision = $macroAveragePrecision\%\n";
# print "Macro-Average Recall = $macroAverageRecall\%\n";
print "Macro-Average F-score = $macroAverageFscore\%\n";

# print "precision0 is $precision0\n";
# print "recall0 is $recall0\n";
# print "precision1 is $precision1\n";
# print "recall1 is $recall1\n";
# print "precision2 is $precision2\n";
# print "recall2 is $recall2\n";


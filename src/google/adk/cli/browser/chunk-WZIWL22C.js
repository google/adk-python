import{m as Ie}from"./chunk-WBLSVR3V.js";import{a as Qe}from"./chunk-GP6TCC26.js";import{M as ce,N as le,R as ue,S as de,T as fe,U as he,V as me,W as ke,X as ye,Y as ot}from"./chunk-QFMJV7VH.js";import{A as we,B as _e,C as De,D as Se,E as Ce,F as Me,G as At,H as Ft,I as Ee,a as yt,c as ae,f as oe,g as c,i as it,k as ge,l as pe,m as ve,n as Te,t as be,u as Et,v as It,w as Lt,x as Yt,y as $t,z as xe}from"./chunk-JRNAXTJ7.js";import"./chunk-EGBSMT36.js";import{e as xt,h as at}from"./chunk-RMXJBC7V.js";var Le=xt((Ot,Wt)=>{"use strict";(function(t,s){typeof Ot=="object"&&typeof Wt<"u"?Wt.exports=s():typeof define=="function"&&define.amd?define(s):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_isoWeek=s()})(Ot,function(){"use strict";var t="day";return function(s,n,a){var r=function(D){return D.add(4-D.isoWeekday(),t)},u=n.prototype;u.isoWeekYear=function(){return r(this).year()},u.isoWeek=function(D){if(!this.$utils().u(D))return this.add(7*(D-this.isoWeek()),t);var S,P,C,W,z=r(this),R=(S=this.isoWeekYear(),P=this.$u,C=(P?a.utc:a)().year(S).startOf("year"),W=4-C.isoWeekday(),C.isoWeekday()>4&&(W+=7),C.add(W,t));return z.diff(R,"week")+1},u.isoWeekday=function(D){return this.$utils().u(D)?this.day()||7:this.day(this.day()%7?D:D-7)};var b=u.startOf;u.startOf=function(D,S){var P=this.$utils(),C=!!P.u(S)||S;return P.p(D)==="isoweek"?C?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):b.bind(this)(D,S)}}})});var Ye=xt((Pt,Vt)=>{"use strict";(function(t,s){typeof Pt=="object"&&typeof Vt<"u"?Vt.exports=s():typeof define=="function"&&define.amd?define(s):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_customParseFormat=s()})(Pt,function(){"use strict";var t={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},s=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,n=/\d/,a=/\d\d/,r=/\d\d?/,u=/\d*[^-_:/,()\s\d]+/,b={},D=function(x){return(x=+x)+(x>68?1900:2e3)},S=function(x){return function(k){this[x]=+k}},P=[/[+-]\d\d:?(\d\d)?|Z/,function(x){(this.zone||(this.zone={})).offset=(function(k){if(!k||k==="Z")return 0;var F=k.match(/([+-]|\d\d)/g),Y=60*F[1]+(+F[2]||0);return Y===0?0:F[0]==="+"?-Y:Y})(x)}],C=function(x){var k=b[x];return k&&(k.indexOf?k:k.s.concat(k.f))},W=function(x,k){var F,Y=b.meridiem;if(Y){for(var G=1;G<=24;G+=1)if(x.indexOf(Y(G,0,k))>-1){F=G>12;break}}else F=x===(k?"pm":"PM");return F},z={A:[u,function(x){this.afternoon=W(x,!1)}],a:[u,function(x){this.afternoon=W(x,!0)}],Q:[n,function(x){this.month=3*(x-1)+1}],S:[n,function(x){this.milliseconds=100*+x}],SS:[a,function(x){this.milliseconds=10*+x}],SSS:[/\d{3}/,function(x){this.milliseconds=+x}],s:[r,S("seconds")],ss:[r,S("seconds")],m:[r,S("minutes")],mm:[r,S("minutes")],H:[r,S("hours")],h:[r,S("hours")],HH:[r,S("hours")],hh:[r,S("hours")],D:[r,S("day")],DD:[a,S("day")],Do:[u,function(x){var k=b.ordinal,F=x.match(/\d+/);if(this.day=F[0],k)for(var Y=1;Y<=31;Y+=1)k(Y).replace(/\[|\]/g,"")===x&&(this.day=Y)}],w:[r,S("week")],ww:[a,S("week")],M:[r,S("month")],MM:[a,S("month")],MMM:[u,function(x){var k=C("months"),F=(C("monthsShort")||k.map(function(Y){return Y.slice(0,3)})).indexOf(x)+1;if(F<1)throw new Error;this.month=F%12||F}],MMMM:[u,function(x){var k=C("months").indexOf(x)+1;if(k<1)throw new Error;this.month=k%12||k}],Y:[/[+-]?\d+/,S("year")],YY:[a,function(x){this.year=D(x)}],YYYY:[/\d{4}/,S("year")],Z:P,ZZ:P};function R(x){var k,F;k=x,F=b&&b.formats;for(var Y=(x=k.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function($,f,y){var g=y&&y.toUpperCase();return f||F[y]||t[y]||F[g].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(T,p,o){return p||o.slice(1)})})).match(s),G=Y.length,X=0;X<G;X+=1){var E=Y[X],v=z[E],d=v&&v[0],M=v&&v[1];Y[X]=M?{regex:d,parser:M}:E.replace(/^\[|\]$/g,"")}return function($){for(var f={},y=0,g=0;y<G;y+=1){var T=Y[y];if(typeof T=="string")g+=T.length;else{var p=T.regex,o=T.parser,l=$.slice(g),h=p.exec(l)[0];o.call(f,h),$=$.replace(h,"")}}return(function(m){var w=m.afternoon;if(w!==void 0){var i=m.hours;w?i<12&&(m.hours+=12):i===12&&(m.hours=0),delete m.afternoon}})(f),f}}return function(x,k,F){F.p.customParseFormat=!0,x&&x.parseTwoDigitYear&&(D=x.parseTwoDigitYear);var Y=k.prototype,G=Y.parse;Y.parse=function(X){var E=X.date,v=X.utc,d=X.args;this.$u=v;var M=d[1];if(typeof M=="string"){var $=d[2]===!0,f=d[3]===!0,y=$||f,g=d[2];f&&(g=d[2]),b=this.$locale(),!$&&g&&(b=F.Ls[g]),this.$d=(function(l,h,m,w){try{if(["x","X"].indexOf(h)>-1)return new Date((h==="X"?1e3:1)*l);var i=R(h)(l),O=i.year,e=i.month,_=i.day,A=i.hours,I=i.minutes,L=i.seconds,H=i.milliseconds,V=i.zone,N=i.week,U=new Date,st=_||(O||e?1:U.getDate()),rt=O||U.getFullYear(),lt=0;O&&!e||(lt=e>0?e-1:U.getMonth());var ut,dt=A||0,j=I||0,nt=L||0,K=H||0;return V?new Date(Date.UTC(rt,lt,st,dt,j,nt,K+60*V.offset*1e3)):m?new Date(Date.UTC(rt,lt,st,dt,j,nt,K)):(ut=new Date(rt,lt,st,dt,j,nt,K),N&&(ut=w(ut).week(N).toDate()),ut)}catch(q){return new Date("")}})(E,M,v,F),this.init(),g&&g!==!0&&(this.$L=this.locale(g).$L),y&&E!=this.format(M)&&(this.$d=new Date("")),b={}}else if(M instanceof Array)for(var T=M.length,p=1;p<=T;p+=1){d[1]=M[p-1];var o=F.apply(this,d);if(o.isValid()){this.$d=o.$d,this.$L=o.$L,this.init();break}p===T&&(this.$d=new Date(""))}else G.call(this,X)}}})});var $e=xt((Nt,Rt)=>{"use strict";(function(t,s){typeof Nt=="object"&&typeof Rt<"u"?Rt.exports=s():typeof define=="function"&&define.amd?define(s):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_advancedFormat=s()})(Nt,function(){"use strict";return function(t,s){var n=s.prototype,a=n.format;n.format=function(r){var u=this,b=this.$locale();if(!this.isValid())return a.bind(this)(r);var D=this.$utils(),S=(r||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function(P){switch(P){case"Q":return Math.ceil((u.$M+1)/3);case"Do":return b.ordinal(u.$D);case"gggg":return u.weekYear();case"GGGG":return u.isoWeekYear();case"wo":return b.ordinal(u.week(),"W");case"w":case"ww":return D.s(u.week(),P==="w"?1:2,"0");case"W":case"WW":return D.s(u.isoWeek(),P==="W"?1:2,"0");case"k":case"kk":return D.s(String(u.$H===0?24:u.$H),P==="k"?1:2,"0");case"X":return Math.floor(u.$d.getTime()/1e3);case"x":return u.$d.getTime();case"z":return"["+u.offsetName()+"]";case"zzz":return"["+u.offsetName("long")+"]";default:return P}});return a.bind(this)(S)}}})});var Ae=xt((zt,Ht)=>{"use strict";(function(t,s){typeof zt=="object"&&typeof Ht<"u"?Ht.exports=s():typeof define=="function"&&define.amd?define(s):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_duration=s()})(zt,function(){"use strict";var t,s,n=1e3,a=6e4,r=36e5,u=864e5,b=/\[([^\]]+)]|Y{1,4}|M{1,4}|D{1,2}|d{1,4}|H{1,2}|h{1,2}|a|A|m{1,2}|s{1,2}|Z{1,2}|SSS/g,D=31536e6,S=2628e6,P=/^(-|\+)?P(?:([-+]?[0-9,.]*)Y)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)W)?(?:([-+]?[0-9,.]*)D)?(?:T(?:([-+]?[0-9,.]*)H)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)S)?)?$/,C={years:D,months:S,days:u,hours:r,minutes:a,seconds:n,milliseconds:1,weeks:6048e5},W=function(E){return E instanceof G},z=function(E,v,d){return new G(E,d,v.$l)},R=function(E){return s.p(E)+"s"},x=function(E){return E<0},k=function(E){return x(E)?Math.ceil(E):Math.floor(E)},F=function(E){return Math.abs(E)},Y=function(E,v){return E?x(E)?{negative:!0,format:""+F(E)+v}:{negative:!1,format:""+E+v}:{negative:!1,format:""}},G=(function(){function E(d,M,$){var f=this;if(this.$d={},this.$l=$,d===void 0&&(this.$ms=0,this.parseFromMilliseconds()),M)return z(d*C[R(M)],this);if(typeof d=="number")return this.$ms=d,this.parseFromMilliseconds(),this;if(typeof d=="object")return Object.keys(d).forEach(function(T){f.$d[R(T)]=d[T]}),this.calMilliseconds(),this;if(typeof d=="string"){var y=d.match(P);if(y){var g=y.slice(2).map(function(T){return T!=null?Number(T):0});return this.$d.years=g[0],this.$d.months=g[1],this.$d.weeks=g[2],this.$d.days=g[3],this.$d.hours=g[4],this.$d.minutes=g[5],this.$d.seconds=g[6],this.calMilliseconds(),this}}return this}var v=E.prototype;return v.calMilliseconds=function(){var d=this;this.$ms=Object.keys(this.$d).reduce(function(M,$){return M+(d.$d[$]||0)*C[$]},0)},v.parseFromMilliseconds=function(){var d=this.$ms;this.$d.years=k(d/D),d%=D,this.$d.months=k(d/S),d%=S,this.$d.days=k(d/u),d%=u,this.$d.hours=k(d/r),d%=r,this.$d.minutes=k(d/a),d%=a,this.$d.seconds=k(d/n),d%=n,this.$d.milliseconds=d},v.toISOString=function(){var d=Y(this.$d.years,"Y"),M=Y(this.$d.months,"M"),$=+this.$d.days||0;this.$d.weeks&&($+=7*this.$d.weeks);var f=Y($,"D"),y=Y(this.$d.hours,"H"),g=Y(this.$d.minutes,"M"),T=this.$d.seconds||0;this.$d.milliseconds&&(T+=this.$d.milliseconds/1e3,T=Math.round(1e3*T)/1e3);var p=Y(T,"S"),o=d.negative||M.negative||f.negative||y.negative||g.negative||p.negative,l=y.format||g.format||p.format?"T":"",h=(o?"-":"")+"P"+d.format+M.format+f.format+l+y.format+g.format+p.format;return h==="P"||h==="-P"?"P0D":h},v.toJSON=function(){return this.toISOString()},v.format=function(d){var M=d||"YYYY-MM-DDTHH:mm:ss",$={Y:this.$d.years,YY:s.s(this.$d.years,2,"0"),YYYY:s.s(this.$d.years,4,"0"),M:this.$d.months,MM:s.s(this.$d.months,2,"0"),D:this.$d.days,DD:s.s(this.$d.days,2,"0"),H:this.$d.hours,HH:s.s(this.$d.hours,2,"0"),m:this.$d.minutes,mm:s.s(this.$d.minutes,2,"0"),s:this.$d.seconds,ss:s.s(this.$d.seconds,2,"0"),SSS:s.s(this.$d.milliseconds,3,"0")};return M.replace(b,function(f,y){return y||String($[f])})},v.as=function(d){return this.$ms/C[R(d)]},v.get=function(d){var M=this.$ms,$=R(d);return $==="milliseconds"?M%=1e3:M=$==="weeks"?k(M/C[$]):this.$d[$],M||0},v.add=function(d,M,$){var f;return f=M?d*C[R(M)]:W(d)?d.$ms:z(d,this).$ms,z(this.$ms+f*($?-1:1),this)},v.subtract=function(d,M){return this.add(d,M,!0)},v.locale=function(d){var M=this.clone();return M.$l=d,M},v.clone=function(){return z(this.$ms,this)},v.humanize=function(d){return t().add(this.$ms,"ms").locale(this.$l).fromNow(!d)},v.valueOf=function(){return this.asMilliseconds()},v.milliseconds=function(){return this.get("milliseconds")},v.asMilliseconds=function(){return this.as("milliseconds")},v.seconds=function(){return this.get("seconds")},v.asSeconds=function(){return this.as("seconds")},v.minutes=function(){return this.get("minutes")},v.asMinutes=function(){return this.as("minutes")},v.hours=function(){return this.get("hours")},v.asHours=function(){return this.as("hours")},v.days=function(){return this.get("days")},v.asDays=function(){return this.as("days")},v.weeks=function(){return this.get("weeks")},v.asWeeks=function(){return this.as("weeks")},v.months=function(){return this.get("months")},v.asMonths=function(){return this.as("months")},v.years=function(){return this.get("years")},v.asYears=function(){return this.as("years")},E})(),X=function(E,v,d){return E.add(v.years()*d,"y").add(v.months()*d,"M").add(v.days()*d,"d").add(v.hours()*d,"h").add(v.minutes()*d,"m").add(v.seconds()*d,"s").add(v.milliseconds()*d,"ms")};return function(E,v,d){t=d,s=d().$utils(),d.duration=function(f,y){var g=d.locale();return z(f,{$l:g},y)},d.isDuration=W;var M=v.prototype.add,$=v.prototype.subtract;v.prototype.add=function(f,y){return W(f)?X(this,f,1):M.bind(this)(f,y)},v.prototype.subtract=function(f,y){return W(f)?X(this,f,-1):$.bind(this)(f,y)}}})});var Pe=at(Qe(),1),Q=at(oe(),1),Ve=at(Le(),1),Ne=at(Ye(),1),Re=at($e(),1),ht=at(oe(),1),Ze=at(Ae(),1);var Bt=(function(){var t=c(function(p,o,l,h){for(l=l||{},h=p.length;h--;l[p[h]]=o);return l},"o"),s=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],n=[1,26],a=[1,27],r=[1,28],u=[1,29],b=[1,30],D=[1,31],S=[1,32],P=[1,33],C=[1,34],W=[1,9],z=[1,10],R=[1,11],x=[1,12],k=[1,13],F=[1,14],Y=[1,15],G=[1,16],X=[1,19],E=[1,20],v=[1,21],d=[1,22],M=[1,23],$=[1,25],f=[1,35],y={trace:c(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:c(function(o,l,h,m,w,i,O){var e=i.length-1;switch(w){case 1:return i[e-1];case 2:this.$=[];break;case 3:i[e-1].push(i[e]),this.$=i[e-1];break;case 4:case 5:this.$=i[e];break;case 6:case 7:this.$=[];break;case 8:m.setWeekday("monday");break;case 9:m.setWeekday("tuesday");break;case 10:m.setWeekday("wednesday");break;case 11:m.setWeekday("thursday");break;case 12:m.setWeekday("friday");break;case 13:m.setWeekday("saturday");break;case 14:m.setWeekday("sunday");break;case 15:m.setWeekend("friday");break;case 16:m.setWeekend("saturday");break;case 17:m.setDateFormat(i[e].substr(11)),this.$=i[e].substr(11);break;case 18:m.enableInclusiveEndDates(),this.$=i[e].substr(18);break;case 19:m.TopAxis(),this.$=i[e].substr(8);break;case 20:m.setAxisFormat(i[e].substr(11)),this.$=i[e].substr(11);break;case 21:m.setTickInterval(i[e].substr(13)),this.$=i[e].substr(13);break;case 22:m.setExcludes(i[e].substr(9)),this.$=i[e].substr(9);break;case 23:m.setIncludes(i[e].substr(9)),this.$=i[e].substr(9);break;case 24:m.setTodayMarker(i[e].substr(12)),this.$=i[e].substr(12);break;case 27:m.setDiagramTitle(i[e].substr(6)),this.$=i[e].substr(6);break;case 28:this.$=i[e].trim(),m.setAccTitle(this.$);break;case 29:case 30:this.$=i[e].trim(),m.setAccDescription(this.$);break;case 31:m.addSection(i[e].substr(8)),this.$=i[e].substr(8);break;case 33:m.addTask(i[e-1],i[e]),this.$="task";break;case 34:this.$=i[e-1],m.setClickEvent(i[e-1],i[e],null);break;case 35:this.$=i[e-2],m.setClickEvent(i[e-2],i[e-1],i[e]);break;case 36:this.$=i[e-2],m.setClickEvent(i[e-2],i[e-1],null),m.setLink(i[e-2],i[e]);break;case 37:this.$=i[e-3],m.setClickEvent(i[e-3],i[e-2],i[e-1]),m.setLink(i[e-3],i[e]);break;case 38:this.$=i[e-2],m.setClickEvent(i[e-2],i[e],null),m.setLink(i[e-2],i[e-1]);break;case 39:this.$=i[e-3],m.setClickEvent(i[e-3],i[e-1],i[e]),m.setLink(i[e-3],i[e-2]);break;case 40:this.$=i[e-1],m.setLink(i[e-1],i[e]);break;case 41:case 47:this.$=i[e-1]+" "+i[e];break;case 42:case 43:case 45:this.$=i[e-2]+" "+i[e-1]+" "+i[e];break;case 44:case 46:this.$=i[e-3]+" "+i[e-2]+" "+i[e-1]+" "+i[e];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(s,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:n,13:a,14:r,15:u,16:b,17:D,18:S,19:18,20:P,21:C,22:W,23:z,24:R,25:x,26:k,27:F,28:Y,29:G,30:X,31:E,33:v,35:d,36:M,37:24,38:$,40:f},t(s,[2,7],{1:[2,1]}),t(s,[2,3]),{9:36,11:17,12:n,13:a,14:r,15:u,16:b,17:D,18:S,19:18,20:P,21:C,22:W,23:z,24:R,25:x,26:k,27:F,28:Y,29:G,30:X,31:E,33:v,35:d,36:M,37:24,38:$,40:f},t(s,[2,5]),t(s,[2,6]),t(s,[2,17]),t(s,[2,18]),t(s,[2,19]),t(s,[2,20]),t(s,[2,21]),t(s,[2,22]),t(s,[2,23]),t(s,[2,24]),t(s,[2,25]),t(s,[2,26]),t(s,[2,27]),{32:[1,37]},{34:[1,38]},t(s,[2,30]),t(s,[2,31]),t(s,[2,32]),{39:[1,39]},t(s,[2,8]),t(s,[2,9]),t(s,[2,10]),t(s,[2,11]),t(s,[2,12]),t(s,[2,13]),t(s,[2,14]),t(s,[2,15]),t(s,[2,16]),{41:[1,40],43:[1,41]},t(s,[2,4]),t(s,[2,28]),t(s,[2,29]),t(s,[2,33]),t(s,[2,34],{42:[1,42],43:[1,43]}),t(s,[2,40],{41:[1,44]}),t(s,[2,35],{43:[1,45]}),t(s,[2,36]),t(s,[2,38],{42:[1,46]}),t(s,[2,37]),t(s,[2,39])],defaultActions:{},parseError:c(function(o,l){if(l.recoverable)this.trace(o);else{var h=new Error(o);throw h.hash=l,h}},"parseError"),parse:c(function(o){var l=this,h=[0],m=[],w=[null],i=[],O=this.table,e="",_=0,A=0,I=0,L=2,H=1,V=i.slice.call(arguments,1),N=Object.create(this.lexer),U={yy:{}};for(var st in this.yy)Object.prototype.hasOwnProperty.call(this.yy,st)&&(U.yy[st]=this.yy[st]);N.setInput(o,U.yy),U.yy.lexer=N,U.yy.parser=this,typeof N.yylloc>"u"&&(N.yylloc={});var rt=N.yylloc;i.push(rt);var lt=N.options&&N.options.ranges;typeof U.yy.parseError=="function"?this.parseError=U.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ut(Z){h.length=h.length-2*Z,w.length=w.length-Z,i.length=i.length-Z}c(ut,"popStack");function dt(){var Z;return Z=m.pop()||N.lex()||H,typeof Z!="number"&&(Z instanceof Array&&(m=Z,Z=m.pop()),Z=l.symbols_[Z]||Z),Z}c(dt,"lex");for(var j,nt,K,q,Ri,Ct,ft={},Tt,tt,ne,bt;;){if(K=h[h.length-1],this.defaultActions[K]?q=this.defaultActions[K]:((j===null||typeof j>"u")&&(j=dt()),q=O[K]&&O[K][j]),typeof q>"u"||!q.length||!q[0]){var Mt="";bt=[];for(Tt in O[K])this.terminals_[Tt]&&Tt>L&&bt.push("'"+this.terminals_[Tt]+"'");N.showPosition?Mt="Parse error on line "+(_+1)+`:
`+N.showPosition()+`
Expecting `+bt.join(", ")+", got '"+(this.terminals_[j]||j)+"'":Mt="Parse error on line "+(_+1)+": Unexpected "+(j==H?"end of input":"'"+(this.terminals_[j]||j)+"'"),this.parseError(Mt,{text:N.match,token:this.terminals_[j]||j,line:N.yylineno,loc:rt,expected:bt})}if(q[0]instanceof Array&&q.length>1)throw new Error("Parse Error: multiple actions possible at state: "+K+", token: "+j);switch(q[0]){case 1:h.push(j),w.push(N.yytext),i.push(N.yylloc),h.push(q[1]),j=null,nt?(j=nt,nt=null):(A=N.yyleng,e=N.yytext,_=N.yylineno,rt=N.yylloc,I>0&&I--);break;case 2:if(tt=this.productions_[q[1]][1],ft.$=w[w.length-tt],ft._$={first_line:i[i.length-(tt||1)].first_line,last_line:i[i.length-1].last_line,first_column:i[i.length-(tt||1)].first_column,last_column:i[i.length-1].last_column},lt&&(ft._$.range=[i[i.length-(tt||1)].range[0],i[i.length-1].range[1]]),Ct=this.performAction.apply(ft,[e,A,_,U.yy,q[1],w,i].concat(V)),typeof Ct<"u")return Ct;tt&&(h=h.slice(0,-1*tt*2),w=w.slice(0,-1*tt),i=i.slice(0,-1*tt)),h.push(this.productions_[q[1]][0]),w.push(ft.$),i.push(ft._$),ne=O[h[h.length-2]][h[h.length-1]],h.push(ne);break;case 3:return!0}}return!0},"parse")},g=(function(){var p={EOF:1,parseError:c(function(l,h){if(this.yy.parser)this.yy.parser.parseError(l,h);else throw new Error(l)},"parseError"),setInput:c(function(o,l){return this.yy=l||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:c(function(){var o=this._input[0];this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o;var l=o.match(/(?:\r\n?|\n).*/g);return l?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:c(function(o){var l=o.length,h=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-l),this.offset-=l;var m=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var w=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===m.length?this.yylloc.first_column:0)+m[m.length-h.length].length-h[0].length:this.yylloc.first_column-l},this.options.ranges&&(this.yylloc.range=[w[0],w[0]+this.yyleng-l]),this.yyleng=this.yytext.length,this},"unput"),more:c(function(){return this._more=!0,this},"more"),reject:c(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:c(function(o){this.unput(this.match.slice(o))},"less"),pastInput:c(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:c(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:c(function(){var o=this.pastInput(),l=new Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+l+"^"},"showPosition"),test_match:c(function(o,l){var h,m,w;if(this.options.backtrack_lexer&&(w={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(w.yylloc.range=this.yylloc.range.slice(0))),m=o[0].match(/(?:\r\n?|\n).*/g),m&&(this.yylineno+=m.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:m?m[m.length-1].length-m[m.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],h=this.performAction.call(this,this.yy,this,l,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var i in w)this[i]=w[i];return!1}return!1},"test_match"),next:c(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var o,l,h,m;this._more||(this.yytext="",this.match="");for(var w=this._currentRules(),i=0;i<w.length;i++)if(h=this._input.match(this.rules[w[i]]),h&&(!l||h[0].length>l[0].length)){if(l=h,m=i,this.options.backtrack_lexer){if(o=this.test_match(h,w[i]),o!==!1)return o;if(this._backtrack){l=!1;continue}else return!1}else if(!this.options.flex)break}return l?(o=this.test_match(l,w[m]),o!==!1?o:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:c(function(){var l=this.next();return l||this.lex()},"lex"),begin:c(function(l){this.conditionStack.push(l)},"begin"),popState:c(function(){var l=this.conditionStack.length-1;return l>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:c(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:c(function(l){return l=this.conditionStack.length-1-Math.abs(l||0),l>=0?this.conditionStack[l]:"INITIAL"},"topState"),pushState:c(function(l){this.begin(l)},"pushState"),stateStackSize:c(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:c(function(l,h,m,w){var i=w;switch(m){case 0:return this.begin("open_directive"),"open_directive";break;case 1:return this.begin("acc_title"),31;break;case 2:return this.popState(),"acc_title_value";break;case 3:return this.begin("acc_descr"),33;break;case 4:return this.popState(),"acc_descr_value";break;case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return p})();y.lexer=g;function T(){this.yy={}}return c(T,"Parser"),T.prototype=y,y.Parser=T,new T})();Bt.parser=Bt;var Ke=Bt;Q.default.extend(Ve.default);Q.default.extend(Ne.default);Q.default.extend(Re.default);var Fe={friday:5,saturday:6},J="",qt="",Zt=void 0,Qt="",gt=[],pt=[],Kt=new Map,Jt=[],Dt=[],kt="",te="",ze=["active","done","crit","milestone","vert"],ee=[],vt=!1,ie=!1,se="sunday",St="saturday",Gt=0,Je=c(function(){Jt=[],Dt=[],kt="",ee=[],wt=0,Ut=void 0,_t=void 0,B=[],J="",qt="",te="",Zt=void 0,Qt="",gt=[],pt=[],vt=!1,ie=!1,Gt=0,Kt=new Map,ue(),se="sunday",St="saturday"},"clear"),ti=c(function(t){qt=t},"setAxisFormat"),ei=c(function(){return qt},"getAxisFormat"),ii=c(function(t){Zt=t},"setTickInterval"),si=c(function(){return Zt},"getTickInterval"),ri=c(function(t){Qt=t},"setTodayMarker"),ni=c(function(){return Qt},"getTodayMarker"),ai=c(function(t){J=t},"setDateFormat"),oi=c(function(){vt=!0},"enableInclusiveEndDates"),ci=c(function(){return vt},"endDatesAreInclusive"),li=c(function(){ie=!0},"enableTopAxis"),ui=c(function(){return ie},"topAxisEnabled"),di=c(function(t){te=t},"setDisplayMode"),fi=c(function(){return te},"getDisplayMode"),hi=c(function(){return J},"getDateFormat"),mi=c(function(t){gt=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),ki=c(function(){return gt},"getIncludes"),yi=c(function(t){pt=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),gi=c(function(){return pt},"getExcludes"),pi=c(function(){return Kt},"getLinks"),vi=c(function(t){kt=t,Jt.push(t)},"addSection"),Ti=c(function(){return Jt},"getSections"),bi=c(function(){let t=Oe(),s=10,n=0;for(;!t&&n<s;)t=Oe(),n++;return Dt=B,Dt},"getTasks"),He=c(function(t,s,n,a){let r=t.format(s.trim()),u=t.format("YYYY-MM-DD");return a.includes(r)||a.includes(u)?!1:n.includes("weekends")&&(t.isoWeekday()===Fe[St]||t.isoWeekday()===Fe[St]+1)||n.includes(t.format("dddd").toLowerCase())?!0:n.includes(r)||n.includes(u)},"isInvalidDate"),xi=c(function(t){se=t},"setWeekday"),wi=c(function(){return se},"getWeekday"),_i=c(function(t){St=t},"setWeekend"),je=c(function(t,s,n,a){if(!n.length||t.manualEndTime)return;let r;t.startTime instanceof Date?r=(0,Q.default)(t.startTime):r=(0,Q.default)(t.startTime,s,!0),r=r.add(1,"d");let u;t.endTime instanceof Date?u=(0,Q.default)(t.endTime):u=(0,Q.default)(t.endTime,s,!0);let[b,D]=Di(r,u,s,n,a);t.endTime=b.toDate(),t.renderEndTime=D},"checkTaskDates"),Di=c(function(t,s,n,a,r){let u=!1,b=null;for(;t<=s;)u||(b=s.toDate()),u=He(t,n,a,r),u&&(s=s.add(1,"d")),t=t.add(1,"d");return[s,b]},"fixTaskDates"),Xt=c(function(t,s,n){if(n=n.trim(),c(D=>{let S=D.trim();return S==="x"||S==="X"},"isTimestampFormat")(s)&&/^\d+$/.test(n))return new Date(Number(n));let u=/^after\s+(?<ids>[\d\w- ]+)/.exec(n);if(u!==null){let D=null;for(let P of u.groups.ids.split(" ")){let C=ct(P);C!==void 0&&(!D||C.endTime>D.endTime)&&(D=C)}if(D)return D.endTime;let S=new Date;return S.setHours(0,0,0,0),S}let b=(0,Q.default)(n,s.trim(),!0);if(b.isValid())return b.toDate();{it.debug("Invalid date:"+n),it.debug("With date format:"+s.trim());let D=new Date(n);if(D===void 0||isNaN(D.getTime())||D.getFullYear()<-1e4||D.getFullYear()>1e4)throw new Error("Invalid date:"+n);return D}},"getStartDate"),Be=c(function(t){let s=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return s!==null?[Number.parseFloat(s[1]),s[2]]:[NaN,"ms"]},"parseDuration"),Ge=c(function(t,s,n,a=!1){n=n.trim();let u=/^until\s+(?<ids>[\d\w- ]+)/.exec(n);if(u!==null){let C=null;for(let z of u.groups.ids.split(" ")){let R=ct(z);R!==void 0&&(!C||R.startTime<C.startTime)&&(C=R)}if(C)return C.startTime;let W=new Date;return W.setHours(0,0,0,0),W}let b=(0,Q.default)(n,s.trim(),!0);if(b.isValid())return a&&(b=b.add(1,"d")),b.toDate();let D=(0,Q.default)(t),[S,P]=Be(n);if(!Number.isNaN(S)){let C=D.add(S,P);C.isValid()&&(D=C)}return D.toDate()},"getEndDate"),wt=0,mt=c(function(t){return t===void 0?(wt=wt+1,"task"+wt):t},"parseId"),Si=c(function(t,s){let n;s.substr(0,1)===":"?n=s.substr(1,s.length):n=s;let a=n.split(","),r={};re(a,r,ze);for(let b=0;b<a.length;b++)a[b]=a[b].trim();let u="";switch(a.length){case 1:r.id=mt(),r.startTime=t.endTime,u=a[0];break;case 2:r.id=mt(),r.startTime=Xt(void 0,J,a[0]),u=a[1];break;case 3:r.id=mt(a[0]),r.startTime=Xt(void 0,J,a[1]),u=a[2];break;default:}return u&&(r.endTime=Ge(r.startTime,J,u,vt),r.manualEndTime=(0,Q.default)(u,"YYYY-MM-DD",!0).isValid(),je(r,J,pt,gt)),r},"compileData"),Ci=c(function(t,s){let n;s.substr(0,1)===":"?n=s.substr(1,s.length):n=s;let a=n.split(","),r={};re(a,r,ze);for(let u=0;u<a.length;u++)a[u]=a[u].trim();switch(a.length){case 1:r.id=mt(),r.startTime={type:"prevTaskEnd",id:t},r.endTime={data:a[0]};break;case 2:r.id=mt(),r.startTime={type:"getStartDate",startData:a[0]},r.endTime={data:a[1]};break;case 3:r.id=mt(a[0]),r.startTime={type:"getStartDate",startData:a[1]},r.endTime={data:a[2]};break;default:}return r},"parseData"),Ut,_t,B=[],Xe={},Mi=c(function(t,s){let n={section:kt,type:kt,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:s},task:t,classes:[]},a=Ci(_t,s);n.raw.startTime=a.startTime,n.raw.endTime=a.endTime,n.id=a.id,n.prevTaskId=_t,n.active=a.active,n.done=a.done,n.crit=a.crit,n.milestone=a.milestone,n.vert=a.vert,n.order=Gt,Gt++;let r=B.push(n);_t=n.id,Xe[n.id]=r-1},"addTask"),ct=c(function(t){let s=Xe[t];return B[s]},"findTaskById"),Ei=c(function(t,s){let n={section:kt,type:kt,description:t,task:t,classes:[]},a=Si(Ut,s);n.startTime=a.startTime,n.endTime=a.endTime,n.id=a.id,n.active=a.active,n.done=a.done,n.crit=a.crit,n.milestone=a.milestone,n.vert=a.vert,Ut=n,Dt.push(n)},"addTaskOrg"),Oe=c(function(){let t=c(function(n){let a=B[n],r="";switch(B[n].raw.startTime.type){case"prevTaskEnd":{let u=ct(a.prevTaskId);a.startTime=u.endTime;break}case"getStartDate":r=Xt(void 0,J,B[n].raw.startTime.startData),r&&(B[n].startTime=r);break}return B[n].startTime&&(B[n].endTime=Ge(B[n].startTime,J,B[n].raw.endTime.data,vt),B[n].endTime&&(B[n].processed=!0,B[n].manualEndTime=(0,Q.default)(B[n].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),je(B[n],J,pt,gt))),B[n].processed},"compileTask"),s=!0;for(let[n,a]of B.entries())t(n),s=s&&a.processed;return s},"compileTasks"),Ii=c(function(t,s){let n=s;ot().securityLevel!=="loose"&&(n=(0,Pe.sanitizeUrl)(s)),t.split(",").forEach(function(a){ct(a)!==void 0&&(qe(a,()=>{window.open(n,"_self")}),Kt.set(a,n))}),Ue(t,"clickable")},"setLink"),Ue=c(function(t,s){t.split(",").forEach(function(n){let a=ct(n);a!==void 0&&a.classes.push(s)})},"setClass"),Li=c(function(t,s,n){if(ot().securityLevel!=="loose"||s===void 0)return;let a=[];if(typeof n=="string"){a=n.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let u=0;u<a.length;u++){let b=a[u].trim();b.startsWith('"')&&b.endsWith('"')&&(b=b.substr(1,b.length-2)),a[u]=b}}a.length===0&&a.push(t),ct(t)!==void 0&&qe(t,()=>{Ie.runFunc(s,...a)})},"setClickFun"),qe=c(function(t,s){ee.push(function(){let n=document.querySelector(`[id="${t}"]`);n!==null&&n.addEventListener("click",function(){s()})},function(){let n=document.querySelector(`[id="${t}-text"]`);n!==null&&n.addEventListener("click",function(){s()})})},"pushFun"),Yi=c(function(t,s,n){t.split(",").forEach(function(a){Li(a,s,n)}),Ue(t,"clickable")},"setClickEvent"),$i=c(function(t){ee.forEach(function(s){s(t)})},"bindFunctions"),Ai={getConfig:c(()=>ot().gantt,"getConfig"),clear:Je,setDateFormat:ai,getDateFormat:hi,enableInclusiveEndDates:oi,endDatesAreInclusive:ci,enableTopAxis:li,topAxisEnabled:ui,setAxisFormat:ti,getAxisFormat:ei,setTickInterval:ii,getTickInterval:si,setTodayMarker:ri,getTodayMarker:ni,setAccTitle:de,getAccTitle:fe,setDiagramTitle:ke,getDiagramTitle:ye,setDisplayMode:di,getDisplayMode:fi,setAccDescription:he,getAccDescription:me,addSection:vi,getSections:Ti,getTasks:bi,addTask:Mi,findTaskById:ct,addTaskOrg:Ei,setIncludes:mi,getIncludes:ki,setExcludes:yi,getExcludes:gi,setClickEvent:Yi,setLink:Ii,getLinks:pi,bindFunctions:$i,parseDuration:Be,isInvalidDate:He,setWeekday:xi,getWeekday:wi,setWeekend:_i};function re(t,s,n){let a=!0;for(;a;)a=!1,n.forEach(function(r){let u="^\\s*"+r+"\\s*$",b=new RegExp(u);t[0].match(b)&&(s[r]=!0,t.shift(1),a=!0)})}c(re,"getTaskTags");ht.default.extend(Ze.default);var Fi=c(function(){it.debug("Something is calling, setConf, remove the call")},"setConf"),We={monday:we,tuesday:_e,wednesday:De,thursday:Se,friday:Ce,saturday:Me,sunday:xe},Oi=c((t,s)=>{let n=[...t].map(()=>-1/0),a=[...t].sort((u,b)=>u.startTime-b.startTime||u.order-b.order),r=0;for(let u of a)for(let b=0;b<n.length;b++)if(u.startTime>=n[b]){n[b]=u.endTime,u.order=b+s,b>r&&(r=b);break}return r},"getMaxIntersections"),et,jt=1e4,Wi=c(function(t,s,n,a){let r=ot().gantt,u=ot().securityLevel,b;u==="sandbox"&&(b=yt("#i"+s));let D=u==="sandbox"?yt(b.nodes()[0].contentDocument.body):yt("body"),S=u==="sandbox"?b.nodes()[0].contentDocument:document,P=S.getElementById(s);et=P.parentElement.offsetWidth,et===void 0&&(et=1200),r.useWidth!==void 0&&(et=r.useWidth);let C=a.db.getTasks(),W=[];for(let f of C)W.push(f.type);W=$(W);let z={},R=2*r.topPadding;if(a.db.getDisplayMode()==="compact"||r.displayMode==="compact"){let f={};for(let g of C)f[g.section]===void 0?f[g.section]=[g]:f[g.section].push(g);let y=0;for(let g of Object.keys(f)){let T=Oi(f[g],y)+1;y+=T,R+=T*(r.barHeight+r.barGap),z[g]=T}}else{R+=C.length*(r.barHeight+r.barGap);for(let f of W)z[f]=C.filter(y=>y.type===f).length}P.setAttribute("viewBox","0 0 "+et+" "+R);let x=D.select(`[id="${s}"]`),k=Ee().domain([pe(C,function(f){return f.startTime}),ge(C,function(f){return f.endTime})]).rangeRound([0,et-r.leftPadding-r.rightPadding]);function F(f,y){let g=f.startTime,T=y.startTime,p=0;return g>T?p=1:g<T&&(p=-1),p}c(F,"taskCompare"),C.sort(F),Y(C,et,R),le(x,R,et,r.useMaxWidth),x.append("text").text(a.db.getDiagramTitle()).attr("x",et/2).attr("y",r.titleTopMargin).attr("class","titleText");function Y(f,y,g){let T=r.barHeight,p=T+r.barGap,o=r.topPadding,l=r.leftPadding,h=be().domain([0,W.length]).range(["#00B9FA","#F95002"]).interpolate(ae);X(p,o,l,y,g,f,a.db.getExcludes(),a.db.getIncludes()),v(l,o,y,g),G(f,p,o,l,T,h,y,g),d(p,o,l,T,h),M(l,o,y,g)}c(Y,"makeGantt");function G(f,y,g,T,p,o,l){f.sort((e,_)=>e.vert===_.vert?0:e.vert?1:-1);let m=[...new Set(f.map(e=>e.order))].map(e=>f.find(_=>_.order===e));x.append("g").selectAll("rect").data(m).enter().append("rect").attr("x",0).attr("y",function(e,_){return _=e.order,_*y+g-2}).attr("width",function(){return l-r.rightPadding/2}).attr("height",y).attr("class",function(e){for(let[_,A]of W.entries())if(e.type===A)return"section section"+_%r.numberSectionStyles;return"section section0"}).enter();let w=x.append("g").selectAll("rect").data(f).enter(),i=a.db.getLinks();if(w.append("rect").attr("id",function(e){return e.id}).attr("rx",3).attr("ry",3).attr("x",function(e){return e.milestone?k(e.startTime)+T+.5*(k(e.endTime)-k(e.startTime))-.5*p:k(e.startTime)+T}).attr("y",function(e,_){return _=e.order,e.vert?r.gridLineStartPadding:_*y+g}).attr("width",function(e){return e.milestone?p:e.vert?.08*p:k(e.renderEndTime||e.endTime)-k(e.startTime)}).attr("height",function(e){return e.vert?C.length*(r.barHeight+r.barGap)+r.barHeight*2:p}).attr("transform-origin",function(e,_){return _=e.order,(k(e.startTime)+T+.5*(k(e.endTime)-k(e.startTime))).toString()+"px "+(_*y+g+.5*p).toString()+"px"}).attr("class",function(e){let _="task",A="";e.classes.length>0&&(A=e.classes.join(" "));let I=0;for(let[H,V]of W.entries())e.type===V&&(I=H%r.numberSectionStyles);let L="";return e.active?e.crit?L+=" activeCrit":L=" active":e.done?e.crit?L=" doneCrit":L=" done":e.crit&&(L+=" crit"),L.length===0&&(L=" task"),e.milestone&&(L=" milestone "+L),e.vert&&(L=" vert "+L),L+=I,L+=" "+A,_+L}),w.append("text").attr("id",function(e){return e.id+"-text"}).text(function(e){return e.task}).attr("font-size",r.fontSize).attr("x",function(e){let _=k(e.startTime),A=k(e.renderEndTime||e.endTime);if(e.milestone&&(_+=.5*(k(e.endTime)-k(e.startTime))-.5*p,A=_+p),e.vert)return k(e.startTime)+T;let I=this.getBBox().width;return I>A-_?A+I+1.5*r.leftPadding>l?_+T-5:A+T+5:(A-_)/2+_+T}).attr("y",function(e,_){return e.vert?r.gridLineStartPadding+C.length*(r.barHeight+r.barGap)+60:(_=e.order,_*y+r.barHeight/2+(r.fontSize/2-2)+g)}).attr("text-height",p).attr("class",function(e){let _=k(e.startTime),A=k(e.endTime);e.milestone&&(A=_+p);let I=this.getBBox().width,L="";e.classes.length>0&&(L=e.classes.join(" "));let H=0;for(let[N,U]of W.entries())e.type===U&&(H=N%r.numberSectionStyles);let V="";return e.active&&(e.crit?V="activeCritText"+H:V="activeText"+H),e.done?e.crit?V=V+" doneCritText"+H:V=V+" doneText"+H:e.crit&&(V=V+" critText"+H),e.milestone&&(V+=" milestoneText"),e.vert&&(V+=" vertText"),I>A-_?A+I+1.5*r.leftPadding>l?L+" taskTextOutsideLeft taskTextOutside"+H+" "+V:L+" taskTextOutsideRight taskTextOutside"+H+" "+V+" width-"+I:L+" taskText taskText"+H+" "+V+" width-"+I}),ot().securityLevel==="sandbox"){let e;e=yt("#i"+s);let _=e.nodes()[0].contentDocument;w.filter(function(A){return i.has(A.id)}).each(function(A){var I=_.querySelector("#"+A.id),L=_.querySelector("#"+A.id+"-text");let H=I.parentNode;var V=_.createElement("a");V.setAttribute("xlink:href",i.get(A.id)),V.setAttribute("target","_top"),H.appendChild(V),V.appendChild(I),V.appendChild(L)})}}c(G,"drawRects");function X(f,y,g,T,p,o,l,h){if(l.length===0&&h.length===0)return;let m,w;for(let{startTime:I,endTime:L}of o)(m===void 0||I<m)&&(m=I),(w===void 0||L>w)&&(w=L);if(!m||!w)return;if((0,ht.default)(w).diff((0,ht.default)(m),"year")>5){it.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}let i=a.db.getDateFormat(),O=[],e=null,_=(0,ht.default)(m);for(;_.valueOf()<=w;)a.db.isInvalidDate(_,i,l,h)?e?e.end=_:e={start:_,end:_}:e&&(O.push(e),e=null),_=_.add(1,"d");x.append("g").selectAll("rect").data(O).enter().append("rect").attr("id",I=>"exclude-"+I.start.format("YYYY-MM-DD")).attr("x",I=>k(I.start.startOf("day"))+g).attr("y",r.gridLineStartPadding).attr("width",I=>k(I.end.endOf("day"))-k(I.start.startOf("day"))).attr("height",p-y-r.gridLineStartPadding).attr("transform-origin",function(I,L){return(k(I.start)+g+.5*(k(I.end)-k(I.start))).toString()+"px "+(L*f+.5*p).toString()+"px"}).attr("class","exclude-range")}c(X,"drawExcludeDays");function E(f,y,g,T){if(g<=0||f>y)return 1/0;let p=y-f,o=ht.default.duration({[T??"day"]:g}).asMilliseconds();return o<=0?1/0:Math.ceil(p/o)}c(E,"getEstimatedTickCount");function v(f,y,g,T){let p=a.db.getDateFormat(),o=a.db.getAxisFormat(),l;o?l=o:p==="D"?l="%d":l=r.axisFormat??"%Y-%m-%d";let h=Te(k).tickSize(-T+y+r.gridLineStartPadding).tickFormat(Ft(l)),w=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(a.db.getTickInterval()||r.tickInterval);if(w!==null){let i=parseInt(w[1],10);if(isNaN(i)||i<=0)it.warn(`Invalid tick interval value: "${w[1]}". Skipping custom tick interval.`);else{let O=w[2],e=a.db.getWeekday()||r.weekday,_=k.domain(),A=_[0],I=_[1],L=E(A,I,i,O);if(L>jt)it.warn(`The tick interval "${i}${O}" would generate ${L} ticks, which exceeds the maximum allowed (${jt}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(O){case"millisecond":h.ticks(Et.every(i));break;case"second":h.ticks(It.every(i));break;case"minute":h.ticks(Lt.every(i));break;case"hour":h.ticks(Yt.every(i));break;case"day":h.ticks($t.every(i));break;case"week":h.ticks(We[e].every(i));break;case"month":h.ticks(At.every(i));break}}}if(x.append("g").attr("class","grid").attr("transform","translate("+f+", "+(T-50)+")").call(h).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),a.db.topAxisEnabled()||r.topAxis){let i=ve(k).tickSize(-T+y+r.gridLineStartPadding).tickFormat(Ft(l));if(w!==null){let O=parseInt(w[1],10);if(isNaN(O)||O<=0)it.warn(`Invalid tick interval value: "${w[1]}". Skipping custom tick interval.`);else{let e=w[2],_=a.db.getWeekday()||r.weekday,A=k.domain(),I=A[0],L=A[1];if(E(I,L,O,e)<=jt)switch(e){case"millisecond":i.ticks(Et.every(O));break;case"second":i.ticks(It.every(O));break;case"minute":i.ticks(Lt.every(O));break;case"hour":i.ticks(Yt.every(O));break;case"day":i.ticks($t.every(O));break;case"week":i.ticks(We[_].every(O));break;case"month":i.ticks(At.every(O));break}}}x.append("g").attr("class","grid").attr("transform","translate("+f+", "+y+")").call(i).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}c(v,"makeGrid");function d(f,y){let g=0,T=Object.keys(z).map(p=>[p,z[p]]);x.append("g").selectAll("text").data(T).enter().append(function(p){let o=p[0].split(ce.lineBreakRegex),l=-(o.length-1)/2,h=S.createElementNS("http://www.w3.org/2000/svg","text");h.setAttribute("dy",l+"em");for(let[m,w]of o.entries()){let i=S.createElementNS("http://www.w3.org/2000/svg","tspan");i.setAttribute("alignment-baseline","central"),i.setAttribute("x","10"),m>0&&i.setAttribute("dy","1em"),i.textContent=w,h.appendChild(i)}return h}).attr("x",10).attr("y",function(p,o){if(o>0)for(let l=0;l<o;l++)return g+=T[o-1][1],p[1]*f/2+g*f+y;else return p[1]*f/2+y}).attr("font-size",r.sectionFontSize).attr("class",function(p){for(let[o,l]of W.entries())if(p[0]===l)return"sectionTitle sectionTitle"+o%r.numberSectionStyles;return"sectionTitle"})}c(d,"vertLabels");function M(f,y,g,T){let p=a.db.getTodayMarker();if(p==="off")return;let o=x.append("g").attr("class","today"),l=new Date,h=o.append("line");h.attr("x1",k(l)+f).attr("x2",k(l)+f).attr("y1",r.titleTopMargin).attr("y2",T-r.titleTopMargin).attr("class","today"),p!==""&&h.attr("style",p.replace(/,/g,";"))}c(M,"drawToday");function $(f){let y={},g=[];for(let T=0,p=f.length;T<p;++T)Object.prototype.hasOwnProperty.call(y,f[T])||(y[f[T]]=!0,g.push(f[T]));return g}c($,"checkUnique")},"draw"),Pi={setConf:Fi,draw:Wi},Vi=c(t=>`
  .mermaid-main-font {
        font-family: ${t.fontFamily};
  }

  .exclude-range {
    fill: ${t.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${t.sectionBkgColor};
  }

  .section2 {
    fill: ${t.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${t.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${t.titleColor};
  }

  .sectionTitle1 {
    fill: ${t.titleColor};
  }

  .sectionTitle2 {
    fill: ${t.titleColor};
  }

  .sectionTitle3 {
    fill: ${t.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${t.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${t.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${t.fontFamily};
    fill: ${t.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${t.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${t.taskTextDarkColor};
    text-anchor: start;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${t.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${t.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${t.taskBkgColor};
    stroke: ${t.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${t.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${t.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${t.activeTaskBkgColor};
    stroke: ${t.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${t.doneTaskBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  /* Done task text displayed outside the bar sits against the diagram background,
     not against the done-task bar, so it must use the outside/contrast color. */
  .doneText0.taskTextOutsideLeft,
  .doneText0.taskTextOutsideRight,
  .doneText1.taskTextOutsideLeft,
  .doneText1.taskTextOutsideRight,
  .doneText2.taskTextOutsideLeft,
  .doneText2.taskTextOutsideRight,
  .doneText3.taskTextOutsideLeft,
  .doneText3.taskTextOutsideRight {
    fill: ${t.taskTextOutsideColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  /* Done-crit task text outside the bar \u2014 same reasoning as doneText above. */
  .doneCritText0.taskTextOutsideLeft,
  .doneCritText0.taskTextOutsideRight,
  .doneCritText1.taskTextOutsideLeft,
  .doneCritText1.taskTextOutsideRight,
  .doneCritText2.taskTextOutsideLeft,
  .doneCritText2.taskTextOutsideRight,
  .doneCritText3.taskTextOutsideLeft,
  .doneCritText3.taskTextOutsideRight {
    fill: ${t.taskTextOutsideColor} !important;
  }

  .vert {
    stroke: ${t.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${t.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.titleColor||t.textColor};
    font-family: ${t.fontFamily};
  }
`,"getStyles"),Ni=Vi,Gi={parser:Ke,db:Ai,renderer:Pi,styles:Ni};export{Gi as diagram};

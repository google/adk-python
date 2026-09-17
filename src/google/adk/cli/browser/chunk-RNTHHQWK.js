import{D as $e}from"./chunk-UKZIEWH5.js";import{a as Ke}from"./chunk-GP6TCC26.js";import{M as le,N as ue,R as de,S as fe,T as he,U as me,V as ke,W as ye,X as ge,Y as ot}from"./chunk-37QI3DOO.js";import{A as _e,B as De,C as Se,D as Ce,E as Me,F as Ee,G as Ft,H as Ot,I as Ie,a as gt,c as oe,f as ce,g as c,i as it,k as pe,l as ve,m as Te,n as be,t as xe,u as It,v as $t,w as Lt,x as Yt,y as At,z as we}from"./chunk-JRNAXTJ7.js";import"./chunk-URMDZFG4.js";import{e as wt,h as at}from"./chunk-RMXJBC7V.js";var Le=wt((Wt,Pt)=>{"use strict";(function(t,i){typeof Wt=="object"&&typeof Pt<"u"?Pt.exports=i():typeof define=="function"&&define.amd?define(i):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_isoWeek=i()})(Wt,function(){"use strict";var t="day";return function(i,a,n){var r=function(D){return D.add(4-D.isoWeekday(),t)},u=a.prototype;u.isoWeekYear=function(){return r(this).year()},u.isoWeek=function(D){if(!this.$utils().u(D))return this.add(7*(D-this.isoWeek()),t);var S,P,C,W,z=r(this),R=(S=this.isoWeekYear(),P=this.$u,C=(P?n.utc:n)().year(S).startOf("year"),W=4-C.isoWeekday(),C.isoWeekday()>4&&(W+=7),C.add(W,t));return z.diff(R,"week")+1},u.isoWeekday=function(D){return this.$utils().u(D)?this.day()||7:this.day(this.day()%7?D:D-7)};var b=u.startOf;u.startOf=function(D,S){var P=this.$utils(),C=!!P.u(S)||S;return P.p(D)==="isoweek"?C?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):b.bind(this)(D,S)}}})});var Ye=wt((Vt,Nt)=>{"use strict";(function(t,i){typeof Vt=="object"&&typeof Nt<"u"?Nt.exports=i():typeof define=="function"&&define.amd?define(i):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_customParseFormat=i()})(Vt,function(){"use strict";var t={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},i=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,a=/\d/,n=/\d\d/,r=/\d\d?/,u=/\d*[^-_:/,()\s\d]+/,b={},D=function(x){return(x=+x)+(x>68?1900:2e3)},S=function(x){return function(k){this[x]=+k}},P=[/[+-]\d\d:?(\d\d)?|Z/,function(x){(this.zone||(this.zone={})).offset=(function(k){if(!k||k==="Z")return 0;var F=k.match(/([+-]|\d\d)/g),L=60*F[1]+(+F[2]||0);return L===0?0:F[0]==="+"?-L:L})(x)}],C=function(x){var k=b[x];return k&&(k.indexOf?k:k.s.concat(k.f))},W=function(x,k){var F,L=b.meridiem;if(L){for(var G=1;G<=24;G+=1)if(x.indexOf(L(G,0,k))>-1){F=G>12;break}}else F=x===(k?"pm":"PM");return F},z={A:[u,function(x){this.afternoon=W(x,!1)}],a:[u,function(x){this.afternoon=W(x,!0)}],Q:[a,function(x){this.month=3*(x-1)+1}],S:[a,function(x){this.milliseconds=100*+x}],SS:[n,function(x){this.milliseconds=10*+x}],SSS:[/\d{3}/,function(x){this.milliseconds=+x}],s:[r,S("seconds")],ss:[r,S("seconds")],m:[r,S("minutes")],mm:[r,S("minutes")],H:[r,S("hours")],h:[r,S("hours")],HH:[r,S("hours")],hh:[r,S("hours")],D:[r,S("day")],DD:[n,S("day")],Do:[u,function(x){var k=b.ordinal,F=x.match(/\d+/);if(this.day=F[0],k)for(var L=1;L<=31;L+=1)k(L).replace(/\[|\]/g,"")===x&&(this.day=L)}],w:[r,S("week")],ww:[n,S("week")],M:[r,S("month")],MM:[n,S("month")],MMM:[u,function(x){var k=C("months"),F=(C("monthsShort")||k.map(function(L){return L.slice(0,3)})).indexOf(x)+1;if(F<1)throw new Error;this.month=F%12||F}],MMMM:[u,function(x){var k=C("months").indexOf(x)+1;if(k<1)throw new Error;this.month=k%12||k}],Y:[/[+-]?\d+/,S("year")],YY:[n,function(x){this.year=D(x)}],YYYY:[/\d{4}/,S("year")],Z:P,ZZ:P};function R(x){var k,F;k=x,F=b&&b.formats;for(var L=(x=k.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function(Y,f,y){var g=y&&y.toUpperCase();return f||F[y]||t[y]||F[g].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(T,p,o){return p||o.slice(1)})})).match(i),G=L.length,X=0;X<G;X+=1){var E=L[X],v=z[E],d=v&&v[0],M=v&&v[1];L[X]=M?{regex:d,parser:M}:E.replace(/^\[|\]$/g,"")}return function(Y){for(var f={},y=0,g=0;y<G;y+=1){var T=L[y];if(typeof T=="string")g+=T.length;else{var p=T.regex,o=T.parser,l=Y.slice(g),h=p.exec(l)[0];o.call(f,h),Y=Y.replace(h,"")}}return(function(m){var w=m.afternoon;if(w!==void 0){var s=m.hours;w?s<12&&(m.hours+=12):s===12&&(m.hours=0),delete m.afternoon}})(f),f}}return function(x,k,F){F.p.customParseFormat=!0,x&&x.parseTwoDigitYear&&(D=x.parseTwoDigitYear);var L=k.prototype,G=L.parse;L.parse=function(X){var E=X.date,v=X.utc,d=X.args;this.$u=v;var M=d[1];if(typeof M=="string"){var Y=d[2]===!0,f=d[3]===!0,y=Y||f,g=d[2];f&&(g=d[2]),b=this.$locale(),!Y&&g&&(b=F.Ls[g]),this.$d=(function(l,h,m,w){try{if(["x","X"].indexOf(h)>-1)return new Date((h==="X"?1e3:1)*l);var s=R(h)(l),O=s.year,e=s.month,_=s.day,A=s.hours,I=s.minutes,$=s.seconds,H=s.milliseconds,V=s.zone,N=s.week,U=new Date,st=_||(O||e?1:U.getDate()),rt=O||U.getFullYear(),lt=0;O&&!e||(lt=e>0?e-1:U.getMonth());var ut,dt=A||0,B=I||0,nt=$||0,K=H||0;return V?new Date(Date.UTC(rt,lt,st,dt,B,nt,K+60*V.offset*1e3)):m?new Date(Date.UTC(rt,lt,st,dt,B,nt,K)):(ut=new Date(rt,lt,st,dt,B,nt,K),N&&(ut=w(ut).week(N).toDate()),ut)}catch(q){return new Date("")}})(E,M,v,F),this.init(),g&&g!==!0&&(this.$L=this.locale(g).$L),y&&E!=this.format(M)&&(this.$d=new Date("")),b={}}else if(M instanceof Array)for(var T=M.length,p=1;p<=T;p+=1){d[1]=M[p-1];var o=F.apply(this,d);if(o.isValid()){this.$d=o.$d,this.$L=o.$L,this.init();break}p===T&&(this.$d=new Date(""))}else G.call(this,X)}}})});var Ae=wt((Rt,zt)=>{"use strict";(function(t,i){typeof Rt=="object"&&typeof zt<"u"?zt.exports=i():typeof define=="function"&&define.amd?define(i):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_advancedFormat=i()})(Rt,function(){"use strict";return function(t,i){var a=i.prototype,n=a.format;a.format=function(r){var u=this,b=this.$locale();if(!this.isValid())return n.bind(this)(r);var D=this.$utils(),S=(r||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function(P){switch(P){case"Q":return Math.ceil((u.$M+1)/3);case"Do":return b.ordinal(u.$D);case"gggg":return u.weekYear();case"GGGG":return u.isoWeekYear();case"wo":return b.ordinal(u.week(),"W");case"w":case"ww":return D.s(u.week(),P==="w"?1:2,"0");case"W":case"WW":return D.s(u.isoWeek(),P==="W"?1:2,"0");case"k":case"kk":return D.s(String(u.$H===0?24:u.$H),P==="k"?1:2,"0");case"X":return Math.floor(u.$d.getTime()/1e3);case"x":return u.$d.getTime();case"z":return"["+u.offsetName()+"]";case"zzz":return"["+u.offsetName("long")+"]";default:return P}});return n.bind(this)(S)}}})});var Fe=wt((Ht,Bt)=>{"use strict";(function(t,i){typeof Ht=="object"&&typeof Bt<"u"?Bt.exports=i():typeof define=="function"&&define.amd?define(i):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_duration=i()})(Ht,function(){"use strict";var t,i,a=1e3,n=6e4,r=36e5,u=864e5,b=/\[([^\]]+)]|Y{1,4}|M{1,4}|D{1,2}|d{1,4}|H{1,2}|h{1,2}|a|A|m{1,2}|s{1,2}|Z{1,2}|SSS/g,D=31536e6,S=2628e6,P=/^(-|\+)?P(?:([-+]?[0-9,.]*)Y)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)W)?(?:([-+]?[0-9,.]*)D)?(?:T(?:([-+]?[0-9,.]*)H)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)S)?)?$/,C={years:D,months:S,days:u,hours:r,minutes:n,seconds:a,milliseconds:1,weeks:6048e5},W=function(E){return E instanceof G},z=function(E,v,d){return new G(E,d,v.$l)},R=function(E){return i.p(E)+"s"},x=function(E){return E<0},k=function(E){return x(E)?Math.ceil(E):Math.floor(E)},F=function(E){return Math.abs(E)},L=function(E,v){return E?x(E)?{negative:!0,format:""+F(E)+v}:{negative:!1,format:""+E+v}:{negative:!1,format:""}},G=(function(){function E(d,M,Y){var f=this;if(this.$d={},this.$l=Y,d===void 0&&(this.$ms=0,this.parseFromMilliseconds()),M)return z(d*C[R(M)],this);if(typeof d=="number")return this.$ms=d,this.parseFromMilliseconds(),this;if(typeof d=="object")return Object.keys(d).forEach(function(T){f.$d[R(T)]=d[T]}),this.calMilliseconds(),this;if(typeof d=="string"){var y=d.match(P);if(y){var g=y.slice(2).map(function(T){return T!=null?Number(T):0});return this.$d.years=g[0],this.$d.months=g[1],this.$d.weeks=g[2],this.$d.days=g[3],this.$d.hours=g[4],this.$d.minutes=g[5],this.$d.seconds=g[6],this.calMilliseconds(),this}}return this}var v=E.prototype;return v.calMilliseconds=function(){var d=this;this.$ms=Object.keys(this.$d).reduce(function(M,Y){return M+(d.$d[Y]||0)*C[Y]},0)},v.parseFromMilliseconds=function(){var d=this.$ms;this.$d.years=k(d/D),d%=D,this.$d.months=k(d/S),d%=S,this.$d.days=k(d/u),d%=u,this.$d.hours=k(d/r),d%=r,this.$d.minutes=k(d/n),d%=n,this.$d.seconds=k(d/a),d%=a,this.$d.milliseconds=d},v.toISOString=function(){var d=L(this.$d.years,"Y"),M=L(this.$d.months,"M"),Y=+this.$d.days||0;this.$d.weeks&&(Y+=7*this.$d.weeks);var f=L(Y,"D"),y=L(this.$d.hours,"H"),g=L(this.$d.minutes,"M"),T=this.$d.seconds||0;this.$d.milliseconds&&(T+=this.$d.milliseconds/1e3,T=Math.round(1e3*T)/1e3);var p=L(T,"S"),o=d.negative||M.negative||f.negative||y.negative||g.negative||p.negative,l=y.format||g.format||p.format?"T":"",h=(o?"-":"")+"P"+d.format+M.format+f.format+l+y.format+g.format+p.format;return h==="P"||h==="-P"?"P0D":h},v.toJSON=function(){return this.toISOString()},v.format=function(d){var M=d||"YYYY-MM-DDTHH:mm:ss",Y={Y:this.$d.years,YY:i.s(this.$d.years,2,"0"),YYYY:i.s(this.$d.years,4,"0"),M:this.$d.months,MM:i.s(this.$d.months,2,"0"),D:this.$d.days,DD:i.s(this.$d.days,2,"0"),H:this.$d.hours,HH:i.s(this.$d.hours,2,"0"),m:this.$d.minutes,mm:i.s(this.$d.minutes,2,"0"),s:this.$d.seconds,ss:i.s(this.$d.seconds,2,"0"),SSS:i.s(this.$d.milliseconds,3,"0")};return M.replace(b,function(f,y){return y||String(Y[f])})},v.as=function(d){return this.$ms/C[R(d)]},v.get=function(d){var M=this.$ms,Y=R(d);return Y==="milliseconds"?M%=1e3:M=Y==="weeks"?k(M/C[Y]):this.$d[Y],M||0},v.add=function(d,M,Y){var f;return f=M?d*C[R(M)]:W(d)?d.$ms:z(d,this).$ms,z(this.$ms+f*(Y?-1:1),this)},v.subtract=function(d,M){return this.add(d,M,!0)},v.locale=function(d){var M=this.clone();return M.$l=d,M},v.clone=function(){return z(this.$ms,this)},v.humanize=function(d){return t().add(this.$ms,"ms").locale(this.$l).fromNow(!d)},v.valueOf=function(){return this.asMilliseconds()},v.milliseconds=function(){return this.get("milliseconds")},v.asMilliseconds=function(){return this.as("milliseconds")},v.seconds=function(){return this.get("seconds")},v.asSeconds=function(){return this.as("seconds")},v.minutes=function(){return this.get("minutes")},v.asMinutes=function(){return this.as("minutes")},v.hours=function(){return this.get("hours")},v.asHours=function(){return this.as("hours")},v.days=function(){return this.get("days")},v.asDays=function(){return this.as("days")},v.weeks=function(){return this.get("weeks")},v.asWeeks=function(){return this.as("weeks")},v.months=function(){return this.get("months")},v.asMonths=function(){return this.as("months")},v.years=function(){return this.get("years")},v.asYears=function(){return this.as("years")},E})(),X=function(E,v,d){return E.add(v.years()*d,"y").add(v.months()*d,"M").add(v.days()*d,"d").add(v.hours()*d,"h").add(v.minutes()*d,"m").add(v.seconds()*d,"s").add(v.milliseconds()*d,"ms")};return function(E,v,d){t=d,i=d().$utils(),d.duration=function(f,y){var g=d.locale();return z(f,{$l:g},y)},d.isDuration=W;var M=v.prototype.add,Y=v.prototype.subtract;v.prototype.add=function(f,y){return W(f)?X(this,f,1):M.bind(this)(f,y)},v.prototype.subtract=function(f,y){return W(f)?X(this,f,-1):Y.bind(this)(f,y)}}})});var Ve=at(Ke(),1),Q=at(ce(),1),Ne=at(Le(),1),Re=at(Ye(),1),ze=at(Ae(),1),mt=at(ce(),1),Qe=at(Fe(),1);var Gt=(function(){var t=c(function(p,o,l,h){for(l=l||{},h=p.length;h--;l[p[h]]=o);return l},"o"),i=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],a=[1,26],n=[1,27],r=[1,28],u=[1,29],b=[1,30],D=[1,31],S=[1,32],P=[1,33],C=[1,34],W=[1,9],z=[1,10],R=[1,11],x=[1,12],k=[1,13],F=[1,14],L=[1,15],G=[1,16],X=[1,19],E=[1,20],v=[1,21],d=[1,22],M=[1,23],Y=[1,25],f=[1,35],y={trace:c(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:c(function(o,l,h,m,w,s,O){var e=s.length-1;switch(w){case 1:return s[e-1];case 2:this.$=[];break;case 3:s[e-1].push(s[e]),this.$=s[e-1];break;case 4:case 5:this.$=s[e];break;case 6:case 7:this.$=[];break;case 8:m.setWeekday("monday");break;case 9:m.setWeekday("tuesday");break;case 10:m.setWeekday("wednesday");break;case 11:m.setWeekday("thursday");break;case 12:m.setWeekday("friday");break;case 13:m.setWeekday("saturday");break;case 14:m.setWeekday("sunday");break;case 15:m.setWeekend("friday");break;case 16:m.setWeekend("saturday");break;case 17:m.setDateFormat(s[e].substr(11)),this.$=s[e].substr(11);break;case 18:m.enableInclusiveEndDates(),this.$=s[e].substr(18);break;case 19:m.TopAxis(),this.$=s[e].substr(8);break;case 20:m.setAxisFormat(s[e].substr(11)),this.$=s[e].substr(11);break;case 21:m.setTickInterval(s[e].substr(13)),this.$=s[e].substr(13);break;case 22:m.setExcludes(s[e].substr(9)),this.$=s[e].substr(9);break;case 23:m.setIncludes(s[e].substr(9)),this.$=s[e].substr(9);break;case 24:m.setTodayMarker(s[e].substr(12)),this.$=s[e].substr(12);break;case 27:m.setDiagramTitle(s[e].substr(6)),this.$=s[e].substr(6);break;case 28:this.$=s[e].trim(),m.setAccTitle(this.$);break;case 29:case 30:this.$=s[e].trim(),m.setAccDescription(this.$);break;case 31:m.addSection(s[e].substr(8)),this.$=s[e].substr(8);break;case 33:m.addTask(s[e-1],s[e]),this.$="task";break;case 34:this.$=s[e-1],m.setClickEvent(s[e-1],s[e],null);break;case 35:this.$=s[e-2],m.setClickEvent(s[e-2],s[e-1],s[e]);break;case 36:this.$=s[e-2],m.setClickEvent(s[e-2],s[e-1],null),m.setLink(s[e-2],s[e]);break;case 37:this.$=s[e-3],m.setClickEvent(s[e-3],s[e-2],s[e-1]),m.setLink(s[e-3],s[e]);break;case 38:this.$=s[e-2],m.setClickEvent(s[e-2],s[e],null),m.setLink(s[e-2],s[e-1]);break;case 39:this.$=s[e-3],m.setClickEvent(s[e-3],s[e-1],s[e]),m.setLink(s[e-3],s[e-2]);break;case 40:this.$=s[e-1],m.setLink(s[e-1],s[e]);break;case 41:case 47:this.$=s[e-1]+" "+s[e];break;case 42:case 43:case 45:this.$=s[e-2]+" "+s[e-1]+" "+s[e];break;case 44:case 46:this.$=s[e-3]+" "+s[e-2]+" "+s[e-1]+" "+s[e];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(i,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:a,13:n,14:r,15:u,16:b,17:D,18:S,19:18,20:P,21:C,22:W,23:z,24:R,25:x,26:k,27:F,28:L,29:G,30:X,31:E,33:v,35:d,36:M,37:24,38:Y,40:f},t(i,[2,7],{1:[2,1]}),t(i,[2,3]),{9:36,11:17,12:a,13:n,14:r,15:u,16:b,17:D,18:S,19:18,20:P,21:C,22:W,23:z,24:R,25:x,26:k,27:F,28:L,29:G,30:X,31:E,33:v,35:d,36:M,37:24,38:Y,40:f},t(i,[2,5]),t(i,[2,6]),t(i,[2,17]),t(i,[2,18]),t(i,[2,19]),t(i,[2,20]),t(i,[2,21]),t(i,[2,22]),t(i,[2,23]),t(i,[2,24]),t(i,[2,25]),t(i,[2,26]),t(i,[2,27]),{32:[1,37]},{34:[1,38]},t(i,[2,30]),t(i,[2,31]),t(i,[2,32]),{39:[1,39]},t(i,[2,8]),t(i,[2,9]),t(i,[2,10]),t(i,[2,11]),t(i,[2,12]),t(i,[2,13]),t(i,[2,14]),t(i,[2,15]),t(i,[2,16]),{41:[1,40],43:[1,41]},t(i,[2,4]),t(i,[2,28]),t(i,[2,29]),t(i,[2,33]),t(i,[2,34],{42:[1,42],43:[1,43]}),t(i,[2,40],{41:[1,44]}),t(i,[2,35],{43:[1,45]}),t(i,[2,36]),t(i,[2,38],{42:[1,46]}),t(i,[2,37]),t(i,[2,39])],defaultActions:{},parseError:c(function(o,l){if(l.recoverable)this.trace(o);else{var h=new Error(o);throw h.hash=l,h}},"parseError"),parse:c(function(o){var l=this,h=[0],m=[],w=[null],s=[],O=this.table,e="",_=0,A=0,I=0,$=2,H=1,V=s.slice.call(arguments,1),N=Object.create(this.lexer),U={yy:{}};for(var st in this.yy)Object.prototype.hasOwnProperty.call(this.yy,st)&&(U.yy[st]=this.yy[st]);N.setInput(o,U.yy),U.yy.lexer=N,U.yy.parser=this,typeof N.yylloc>"u"&&(N.yylloc={});var rt=N.yylloc;s.push(rt);var lt=N.options&&N.options.ranges;typeof U.yy.parseError=="function"?this.parseError=U.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ut(Z){h.length=h.length-2*Z,w.length=w.length-Z,s.length=s.length-Z}c(ut,"popStack");function dt(){var Z;return Z=m.pop()||N.lex()||H,typeof Z!="number"&&(Z instanceof Array&&(m=Z,Z=m.pop()),Z=l.symbols_[Z]||Z),Z}c(dt,"lex");for(var B,nt,K,q,Hi,Mt,ft={},bt,tt,ae,xt;;){if(K=h[h.length-1],this.defaultActions[K]?q=this.defaultActions[K]:((B===null||typeof B>"u")&&(B=dt()),q=O[K]&&O[K][B]),typeof q>"u"||!q.length||!q[0]){var Et="";xt=[];for(bt in O[K])this.terminals_[bt]&&bt>$&&xt.push("'"+this.terminals_[bt]+"'");N.showPosition?Et="Parse error on line "+(_+1)+`:
`+N.showPosition()+`
Expecting `+xt.join(", ")+", got '"+(this.terminals_[B]||B)+"'":Et="Parse error on line "+(_+1)+": Unexpected "+(B==H?"end of input":"'"+(this.terminals_[B]||B)+"'"),this.parseError(Et,{text:N.match,token:this.terminals_[B]||B,line:N.yylineno,loc:rt,expected:xt})}if(q[0]instanceof Array&&q.length>1)throw new Error("Parse Error: multiple actions possible at state: "+K+", token: "+B);switch(q[0]){case 1:h.push(B),w.push(N.yytext),s.push(N.yylloc),h.push(q[1]),B=null,nt?(B=nt,nt=null):(A=N.yyleng,e=N.yytext,_=N.yylineno,rt=N.yylloc,I>0&&I--);break;case 2:if(tt=this.productions_[q[1]][1],ft.$=w[w.length-tt],ft._$={first_line:s[s.length-(tt||1)].first_line,last_line:s[s.length-1].last_line,first_column:s[s.length-(tt||1)].first_column,last_column:s[s.length-1].last_column},lt&&(ft._$.range=[s[s.length-(tt||1)].range[0],s[s.length-1].range[1]]),Mt=this.performAction.apply(ft,[e,A,_,U.yy,q[1],w,s].concat(V)),typeof Mt<"u")return Mt;tt&&(h=h.slice(0,-1*tt*2),w=w.slice(0,-1*tt),s=s.slice(0,-1*tt)),h.push(this.productions_[q[1]][0]),w.push(ft.$),s.push(ft._$),ae=O[h[h.length-2]][h[h.length-1]],h.push(ae);break;case 3:return!0}}return!0},"parse")},g=(function(){var p={EOF:1,parseError:c(function(l,h){if(this.yy.parser)this.yy.parser.parseError(l,h);else throw new Error(l)},"parseError"),setInput:c(function(o,l){return this.yy=l||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:c(function(){var o=this._input[0];this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o;var l=o.match(/(?:\r\n?|\n).*/g);return l?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:c(function(o){var l=o.length,h=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-l),this.offset-=l;var m=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var w=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===m.length?this.yylloc.first_column:0)+m[m.length-h.length].length-h[0].length:this.yylloc.first_column-l},this.options.ranges&&(this.yylloc.range=[w[0],w[0]+this.yyleng-l]),this.yyleng=this.yytext.length,this},"unput"),more:c(function(){return this._more=!0,this},"more"),reject:c(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:c(function(o){this.unput(this.match.slice(o))},"less"),pastInput:c(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:c(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:c(function(){var o=this.pastInput(),l=new Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+l+"^"},"showPosition"),test_match:c(function(o,l){var h,m,w;if(this.options.backtrack_lexer&&(w={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(w.yylloc.range=this.yylloc.range.slice(0))),m=o[0].match(/(?:\r\n?|\n).*/g),m&&(this.yylineno+=m.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:m?m[m.length-1].length-m[m.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],h=this.performAction.call(this,this.yy,this,l,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var s in w)this[s]=w[s];return!1}return!1},"test_match"),next:c(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var o,l,h,m;this._more||(this.yytext="",this.match="");for(var w=this._currentRules(),s=0;s<w.length;s++)if(h=this._input.match(this.rules[w[s]]),h&&(!l||h[0].length>l[0].length)){if(l=h,m=s,this.options.backtrack_lexer){if(o=this.test_match(h,w[s]),o!==!1)return o;if(this._backtrack){l=!1;continue}else return!1}else if(!this.options.flex)break}return l?(o=this.test_match(l,w[m]),o!==!1?o:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:c(function(){var l=this.next();return l||this.lex()},"lex"),begin:c(function(l){this.conditionStack.push(l)},"begin"),popState:c(function(){var l=this.conditionStack.length-1;return l>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:c(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:c(function(l){return l=this.conditionStack.length-1-Math.abs(l||0),l>=0?this.conditionStack[l]:"INITIAL"},"topState"),pushState:c(function(l){this.begin(l)},"pushState"),stateStackSize:c(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:c(function(l,h,m,w){var s=w;switch(m){case 0:return this.begin("open_directive"),"open_directive";break;case 1:return this.begin("acc_title"),31;break;case 2:return this.popState(),"acc_title_value";break;case 3:return this.begin("acc_descr"),33;break;case 4:return this.popState(),"acc_descr_value";break;case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return p})();y.lexer=g;function T(){this.yy={}}return c(T,"Parser"),T.prototype=y,y.Parser=T,new T})();Gt.parser=Gt;var Je=Gt;Q.default.extend(Ne.default);Q.default.extend(Re.default);Q.default.extend(ze.default);var Oe={friday:5,saturday:6},J="",Zt="",Qt=void 0,Kt="",pt=[],vt=[],Jt=new Map,te=[],St=[],yt="",ee="",He=["active","done","crit","milestone","vert"],ie=[],ht="",Tt=!1,se=!1,re="sunday",Ct="saturday",Xt=0,ti=c(function(){te=[],St=[],yt="",ie=[],_t=0,qt=void 0,Dt=void 0,j=[],J="",Zt="",ee="",Qt=void 0,Kt="",pt=[],vt=[],Tt=!1,se=!1,Xt=0,Jt=new Map,ht="",de(),re="sunday",Ct="saturday"},"clear"),ei=c(function(t){ht=t},"setDiagramId"),ii=c(function(t){Zt=t},"setAxisFormat"),si=c(function(){return Zt},"getAxisFormat"),ri=c(function(t){Qt=t},"setTickInterval"),ni=c(function(){return Qt},"getTickInterval"),ai=c(function(t){Kt=t},"setTodayMarker"),oi=c(function(){return Kt},"getTodayMarker"),ci=c(function(t){J=t},"setDateFormat"),li=c(function(){Tt=!0},"enableInclusiveEndDates"),ui=c(function(){return Tt},"endDatesAreInclusive"),di=c(function(){se=!0},"enableTopAxis"),fi=c(function(){return se},"topAxisEnabled"),hi=c(function(t){ee=t},"setDisplayMode"),mi=c(function(){return ee},"getDisplayMode"),ki=c(function(){return J},"getDateFormat"),yi=c(function(t){pt=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),gi=c(function(){return pt},"getIncludes"),pi=c(function(t){vt=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),vi=c(function(){return vt},"getExcludes"),Ti=c(function(){return Jt},"getLinks"),bi=c(function(t){yt=t,te.push(t)},"addSection"),xi=c(function(){return te},"getSections"),wi=c(function(){let t=We(),i=10,a=0;for(;!t&&a<i;)t=We(),a++;return St=j,St},"getTasks"),Be=c(function(t,i,a,n){let r=t.format(i.trim()),u=t.format("YYYY-MM-DD");return n.includes(r)||n.includes(u)?!1:a.includes("weekends")&&(t.isoWeekday()===Oe[Ct]||t.isoWeekday()===Oe[Ct]+1)||a.includes(t.format("dddd").toLowerCase())?!0:a.includes(r)||a.includes(u)},"isInvalidDate"),_i=c(function(t){re=t},"setWeekday"),Di=c(function(){return re},"getWeekday"),Si=c(function(t){Ct=t},"setWeekend"),je=c(function(t,i,a,n){if(!a.length||t.manualEndTime)return;let r;t.startTime instanceof Date?r=(0,Q.default)(t.startTime):r=(0,Q.default)(t.startTime,i,!0),r=r.add(1,"d");let u;t.endTime instanceof Date?u=(0,Q.default)(t.endTime):u=(0,Q.default)(t.endTime,i,!0);let[b,D]=Ci(r,u,i,a,n);t.endTime=b.toDate(),t.renderEndTime=D},"checkTaskDates"),Ci=c(function(t,i,a,n,r){let u=!1,b=null;for(;t<=i;)u||(b=i.toDate()),u=Be(t,a,n,r),u&&(i=i.add(1,"d")),t=t.add(1,"d");return[i,b]},"fixTaskDates"),Ut=c(function(t,i,a){if(a=a.trim(),c(D=>{let S=D.trim();return S==="x"||S==="X"},"isTimestampFormat")(i)&&/^\d+$/.test(a))return new Date(Number(a));let u=/^after\s+(?<ids>[\d\w- ]+)/.exec(a);if(u!==null){let D=null;for(let P of u.groups.ids.split(" ")){let C=ct(P);C!==void 0&&(!D||C.endTime>D.endTime)&&(D=C)}if(D)return D.endTime;let S=new Date;return S.setHours(0,0,0,0),S}let b=(0,Q.default)(a,i.trim(),!0);if(b.isValid())return b.toDate();{it.debug("Invalid date:"+a),it.debug("With date format:"+i.trim());let D=new Date(a);if(D===void 0||isNaN(D.getTime())||D.getFullYear()<-1e4||D.getFullYear()>1e4)throw new Error("Invalid date:"+a);return D}},"getStartDate"),Ge=c(function(t){let i=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return i!==null?[Number.parseFloat(i[1]),i[2]]:[NaN,"ms"]},"parseDuration"),Xe=c(function(t,i,a,n=!1){a=a.trim();let u=/^until\s+(?<ids>[\d\w- ]+)/.exec(a);if(u!==null){let C=null;for(let z of u.groups.ids.split(" ")){let R=ct(z);R!==void 0&&(!C||R.startTime<C.startTime)&&(C=R)}if(C)return C.startTime;let W=new Date;return W.setHours(0,0,0,0),W}let b=(0,Q.default)(a,i.trim(),!0);if(b.isValid())return n&&(b=b.add(1,"d")),b.toDate();let D=(0,Q.default)(t),[S,P]=Ge(a);if(!Number.isNaN(S)){let C=D.add(S,P);C.isValid()&&(D=C)}return D.toDate()},"getEndDate"),_t=0,kt=c(function(t){return t===void 0?(_t=_t+1,"task"+_t):t},"parseId"),Mi=c(function(t,i){let a;i.substr(0,1)===":"?a=i.substr(1,i.length):a=i;let n=a.split(","),r={};ne(n,r,He);for(let b=0;b<n.length;b++)n[b]=n[b].trim();let u="";switch(n.length){case 1:r.id=kt(),r.startTime=t.endTime,u=n[0];break;case 2:r.id=kt(),r.startTime=Ut(void 0,J,n[0]),u=n[1];break;case 3:r.id=kt(n[0]),r.startTime=Ut(void 0,J,n[1]),u=n[2];break;default:}return u&&(r.endTime=Xe(r.startTime,J,u,Tt),r.manualEndTime=(0,Q.default)(u,"YYYY-MM-DD",!0).isValid(),je(r,J,vt,pt)),r},"compileData"),Ei=c(function(t,i){let a;i.substr(0,1)===":"?a=i.substr(1,i.length):a=i;let n=a.split(","),r={};ne(n,r,He);for(let u=0;u<n.length;u++)n[u]=n[u].trim();switch(n.length){case 1:r.id=kt(),r.startTime={type:"prevTaskEnd",id:t},r.endTime={data:n[0]};break;case 2:r.id=kt(),r.startTime={type:"getStartDate",startData:n[0]},r.endTime={data:n[1]};break;case 3:r.id=kt(n[0]),r.startTime={type:"getStartDate",startData:n[1]},r.endTime={data:n[2]};break;default:}return r},"parseData"),qt,Dt,j=[],Ue={},Ii=c(function(t,i){let a={section:yt,type:yt,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:i},task:t,classes:[]},n=Ei(Dt,i);a.raw.startTime=n.startTime,a.raw.endTime=n.endTime,a.id=n.id,a.prevTaskId=Dt,a.active=n.active,a.done=n.done,a.crit=n.crit,a.milestone=n.milestone,a.vert=n.vert,a.order=Xt,Xt++;let r=j.push(a);Dt=a.id,Ue[a.id]=r-1},"addTask"),ct=c(function(t){let i=Ue[t];return j[i]},"findTaskById"),$i=c(function(t,i){let a={section:yt,type:yt,description:t,task:t,classes:[]},n=Mi(qt,i);a.startTime=n.startTime,a.endTime=n.endTime,a.id=n.id,a.active=n.active,a.done=n.done,a.crit=n.crit,a.milestone=n.milestone,a.vert=n.vert,qt=a,St.push(a)},"addTaskOrg"),We=c(function(){let t=c(function(a){let n=j[a],r="";switch(j[a].raw.startTime.type){case"prevTaskEnd":{let u=ct(n.prevTaskId);n.startTime=u.endTime;break}case"getStartDate":r=Ut(void 0,J,j[a].raw.startTime.startData),r&&(j[a].startTime=r);break}return j[a].startTime&&(j[a].endTime=Xe(j[a].startTime,J,j[a].raw.endTime.data,Tt),j[a].endTime&&(j[a].processed=!0,j[a].manualEndTime=(0,Q.default)(j[a].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),je(j[a],J,vt,pt))),j[a].processed},"compileTask"),i=!0;for(let[a,n]of j.entries())t(a),i=i&&n.processed;return i},"compileTasks"),Li=c(function(t,i){let a=i;ot().securityLevel!=="loose"&&(a=(0,Ve.sanitizeUrl)(i)),t.split(",").forEach(function(n){ct(n)!==void 0&&(Ze(n,()=>{window.open(a,"_self")}),Jt.set(n,a))}),qe(t,"clickable")},"setLink"),qe=c(function(t,i){t.split(",").forEach(function(a){let n=ct(a);n!==void 0&&n.classes.push(i)})},"setClass"),Yi=c(function(t,i,a){if(ot().securityLevel!=="loose"||i===void 0)return;let n=[];if(typeof a=="string"){n=a.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let u=0;u<n.length;u++){let b=n[u].trim();b.startsWith('"')&&b.endsWith('"')&&(b=b.substr(1,b.length-2)),n[u]=b}}n.length===0&&n.push(t),ct(t)!==void 0&&Ze(t,()=>{$e.runFunc(i,...n)})},"setClickFun"),Ze=c(function(t,i){ie.push(function(){let a=ht?`${ht}-${t}`:t,n=document.querySelector(`[id="${a}"]`);n!==null&&n.addEventListener("click",function(){i()})},function(){let a=ht?`${ht}-${t}`:t,n=document.querySelector(`[id="${a}-text"]`);n!==null&&n.addEventListener("click",function(){i()})})},"pushFun"),Ai=c(function(t,i,a){t.split(",").forEach(function(n){Yi(n,i,a)}),qe(t,"clickable")},"setClickEvent"),Fi=c(function(t){ie.forEach(function(i){i(t)})},"bindFunctions"),Oi={getConfig:c(()=>ot().gantt,"getConfig"),clear:ti,setDateFormat:ci,getDateFormat:ki,enableInclusiveEndDates:li,endDatesAreInclusive:ui,enableTopAxis:di,topAxisEnabled:fi,setAxisFormat:ii,getAxisFormat:si,setTickInterval:ri,getTickInterval:ni,setTodayMarker:ai,getTodayMarker:oi,setAccTitle:fe,getAccTitle:he,setDiagramTitle:ye,getDiagramTitle:ge,setDiagramId:ei,setDisplayMode:hi,getDisplayMode:mi,setAccDescription:me,getAccDescription:ke,addSection:bi,getSections:xi,getTasks:wi,addTask:Ii,findTaskById:ct,addTaskOrg:$i,setIncludes:yi,getIncludes:gi,setExcludes:pi,getExcludes:vi,setClickEvent:Ai,setLink:Li,getLinks:Ti,bindFunctions:Fi,parseDuration:Ge,isInvalidDate:Be,setWeekday:_i,getWeekday:Di,setWeekend:Si};function ne(t,i,a){let n=!0;for(;n;)n=!1,a.forEach(function(r){let u="^\\s*"+r+"\\s*$",b=new RegExp(u);t[0].match(b)&&(i[r]=!0,t.shift(1),n=!0)})}c(ne,"getTaskTags");mt.default.extend(Qe.default);var Wi=c(function(){it.debug("Something is calling, setConf, remove the call")},"setConf"),Pe={monday:_e,tuesday:De,wednesday:Se,thursday:Ce,friday:Me,saturday:Ee,sunday:we},Pi=c((t,i)=>{let a=[...t].map(()=>-1/0),n=[...t].sort((u,b)=>u.startTime-b.startTime||u.order-b.order),r=0;for(let u of n)for(let b=0;b<a.length;b++)if(u.startTime>=a[b]){a[b]=u.endTime,u.order=b+i,b>r&&(r=b);break}return r},"getMaxIntersections"),et,jt=1e4,Vi=c(function(t,i,a,n){let r=ot().gantt;n.db.setDiagramId(i);let u=ot().securityLevel,b;u==="sandbox"&&(b=gt("#i"+i));let D=u==="sandbox"?gt(b.nodes()[0].contentDocument.body):gt("body"),S=u==="sandbox"?b.nodes()[0].contentDocument:document,P=S.getElementById(i);et=P.parentElement.offsetWidth,et===void 0&&(et=1200),r.useWidth!==void 0&&(et=r.useWidth);let C=n.db.getTasks(),W=[];for(let f of C)W.push(f.type);W=Y(W);let z={},R=2*r.topPadding;if(n.db.getDisplayMode()==="compact"||r.displayMode==="compact"){let f={};for(let g of C)f[g.section]===void 0?f[g.section]=[g]:f[g.section].push(g);let y=0;for(let g of Object.keys(f)){let T=Pi(f[g],y)+1;y+=T,R+=T*(r.barHeight+r.barGap),z[g]=T}}else{R+=C.length*(r.barHeight+r.barGap);for(let f of W)z[f]=C.filter(y=>y.type===f).length}P.setAttribute("viewBox","0 0 "+et+" "+R);let x=D.select(`[id="${i}"]`),k=Ie().domain([ve(C,function(f){return f.startTime}),pe(C,function(f){return f.endTime})]).rangeRound([0,et-r.leftPadding-r.rightPadding]);function F(f,y){let g=f.startTime,T=y.startTime,p=0;return g>T?p=1:g<T&&(p=-1),p}c(F,"taskCompare"),C.sort(F),L(C,et,R),ue(x,R,et,r.useMaxWidth),x.append("text").text(n.db.getDiagramTitle()).attr("x",et/2).attr("y",r.titleTopMargin).attr("class","titleText");function L(f,y,g){let T=r.barHeight,p=T+r.barGap,o=r.topPadding,l=r.leftPadding,h=xe().domain([0,W.length]).range(["#00B9FA","#F95002"]).interpolate(oe);X(p,o,l,y,g,f,n.db.getExcludes(),n.db.getIncludes()),v(l,o,y,g),G(f,p,o,l,T,h,y,g),d(p,o,l,T,h),M(l,o,y,g)}c(L,"makeGantt");function G(f,y,g,T,p,o,l){f.sort((e,_)=>e.vert===_.vert?0:e.vert?1:-1);let m=[...new Set(f.map(e=>e.order))].map(e=>f.find(_=>_.order===e));x.append("g").selectAll("rect").data(m).enter().append("rect").attr("x",0).attr("y",function(e,_){return _=e.order,_*y+g-2}).attr("width",function(){return l-r.rightPadding/2}).attr("height",y).attr("class",function(e){for(let[_,A]of W.entries())if(e.type===A)return"section section"+_%r.numberSectionStyles;return"section section0"}).enter();let w=x.append("g").selectAll("rect").data(f).enter(),s=n.db.getLinks();if(w.append("rect").attr("id",function(e){return i+"-"+e.id}).attr("rx",3).attr("ry",3).attr("x",function(e){return e.milestone?k(e.startTime)+T+.5*(k(e.endTime)-k(e.startTime))-.5*p:k(e.startTime)+T}).attr("y",function(e,_){return _=e.order,e.vert?r.gridLineStartPadding:_*y+g}).attr("width",function(e){return e.milestone?p:e.vert?.08*p:k(e.renderEndTime||e.endTime)-k(e.startTime)}).attr("height",function(e){return e.vert?C.length*(r.barHeight+r.barGap)+r.barHeight*2:p}).attr("transform-origin",function(e,_){return _=e.order,(k(e.startTime)+T+.5*(k(e.endTime)-k(e.startTime))).toString()+"px "+(_*y+g+.5*p).toString()+"px"}).attr("class",function(e){let _="task",A="";e.classes.length>0&&(A=e.classes.join(" "));let I=0;for(let[H,V]of W.entries())e.type===V&&(I=H%r.numberSectionStyles);let $="";return e.active?e.crit?$+=" activeCrit":$=" active":e.done?e.crit?$=" doneCrit":$=" done":e.crit&&($+=" crit"),$.length===0&&($=" task"),e.milestone&&($=" milestone "+$),e.vert&&($=" vert "+$),$+=I,$+=" "+A,_+$}),w.append("text").attr("id",function(e){return i+"-"+e.id+"-text"}).text(function(e){return e.task}).attr("font-size",r.fontSize).attr("x",function(e){let _=k(e.startTime),A=k(e.renderEndTime||e.endTime);if(e.milestone&&(_+=.5*(k(e.endTime)-k(e.startTime))-.5*p,A=_+p),e.vert)return k(e.startTime)+T;let I=this.getBBox().width;return I>A-_?A+I+1.5*r.leftPadding>l?_+T-5:A+T+5:(A-_)/2+_+T}).attr("y",function(e,_){return e.vert?r.gridLineStartPadding+C.length*(r.barHeight+r.barGap)+60:(_=e.order,_*y+r.barHeight/2+(r.fontSize/2-2)+g)}).attr("text-height",p).attr("class",function(e){let _=k(e.startTime),A=k(e.endTime);e.milestone&&(A=_+p);let I=this.getBBox().width,$="";e.classes.length>0&&($=e.classes.join(" "));let H=0;for(let[N,U]of W.entries())e.type===U&&(H=N%r.numberSectionStyles);let V="";return e.active&&(e.crit?V="activeCritText"+H:V="activeText"+H),e.done?e.crit?V=V+" doneCritText"+H:V=V+" doneText"+H:e.crit&&(V=V+" critText"+H),e.milestone&&(V+=" milestoneText"),e.vert&&(V+=" vertText"),I>A-_?A+I+1.5*r.leftPadding>l?$+" taskTextOutsideLeft taskTextOutside"+H+" "+V:$+" taskTextOutsideRight taskTextOutside"+H+" "+V+" width-"+I:$+" taskText taskText"+H+" "+V+" width-"+I}),ot().securityLevel==="sandbox"){let e;e=gt("#i"+i);let _=e.nodes()[0].contentDocument;w.filter(function(A){return s.has(A.id)}).each(function(A){var I=_.querySelector("#"+CSS.escape(i+"-"+A.id)),$=_.querySelector("#"+CSS.escape(i+"-"+A.id+"-text"));let H=I.parentNode;var V=_.createElement("a");V.setAttribute("xlink:href",s.get(A.id)),V.setAttribute("target","_top"),H.appendChild(V),V.appendChild(I),V.appendChild($)})}}c(G,"drawRects");function X(f,y,g,T,p,o,l,h){if(l.length===0&&h.length===0)return;let m,w;for(let{startTime:I,endTime:$}of o)(m===void 0||I<m)&&(m=I),(w===void 0||$>w)&&(w=$);if(!m||!w)return;if((0,mt.default)(w).diff((0,mt.default)(m),"year")>5){it.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}let s=n.db.getDateFormat(),O=[],e=null,_=(0,mt.default)(m);for(;_.valueOf()<=w;)n.db.isInvalidDate(_,s,l,h)?e?e.end=_:e={start:_,end:_}:e&&(O.push(e),e=null),_=_.add(1,"d");x.append("g").selectAll("rect").data(O).enter().append("rect").attr("id",I=>i+"-exclude-"+I.start.format("YYYY-MM-DD")).attr("x",I=>k(I.start.startOf("day"))+g).attr("y",r.gridLineStartPadding).attr("width",I=>k(I.end.endOf("day"))-k(I.start.startOf("day"))).attr("height",p-y-r.gridLineStartPadding).attr("transform-origin",function(I,$){return(k(I.start)+g+.5*(k(I.end)-k(I.start))).toString()+"px "+($*f+.5*p).toString()+"px"}).attr("class","exclude-range")}c(X,"drawExcludeDays");function E(f,y,g,T){if(g<=0||f>y)return 1/0;let p=y-f,o=mt.default.duration({[T??"day"]:g}).asMilliseconds();return o<=0?1/0:Math.ceil(p/o)}c(E,"getEstimatedTickCount");function v(f,y,g,T){let p=n.db.getDateFormat(),o=n.db.getAxisFormat(),l;o?l=o:p==="D"?l="%d":l=r.axisFormat??"%Y-%m-%d";let h=be(k).tickSize(-T+y+r.gridLineStartPadding).tickFormat(Ot(l)),w=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(n.db.getTickInterval()||r.tickInterval);if(w!==null){let s=parseInt(w[1],10);if(isNaN(s)||s<=0)it.warn(`Invalid tick interval value: "${w[1]}". Skipping custom tick interval.`);else{let O=w[2],e=n.db.getWeekday()||r.weekday,_=k.domain(),A=_[0],I=_[1],$=E(A,I,s,O);if($>jt)it.warn(`The tick interval "${s}${O}" would generate ${$} ticks, which exceeds the maximum allowed (${jt}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(O){case"millisecond":h.ticks(It.every(s));break;case"second":h.ticks($t.every(s));break;case"minute":h.ticks(Lt.every(s));break;case"hour":h.ticks(Yt.every(s));break;case"day":h.ticks(At.every(s));break;case"week":h.ticks(Pe[e].every(s));break;case"month":h.ticks(Ft.every(s));break}}}if(x.append("g").attr("class","grid").attr("transform","translate("+f+", "+(T-50)+")").call(h).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),n.db.topAxisEnabled()||r.topAxis){let s=Te(k).tickSize(-T+y+r.gridLineStartPadding).tickFormat(Ot(l));if(w!==null){let O=parseInt(w[1],10);if(isNaN(O)||O<=0)it.warn(`Invalid tick interval value: "${w[1]}". Skipping custom tick interval.`);else{let e=w[2],_=n.db.getWeekday()||r.weekday,A=k.domain(),I=A[0],$=A[1];if(E(I,$,O,e)<=jt)switch(e){case"millisecond":s.ticks(It.every(O));break;case"second":s.ticks($t.every(O));break;case"minute":s.ticks(Lt.every(O));break;case"hour":s.ticks(Yt.every(O));break;case"day":s.ticks(At.every(O));break;case"week":s.ticks(Pe[_].every(O));break;case"month":s.ticks(Ft.every(O));break}}}x.append("g").attr("class","grid").attr("transform","translate("+f+", "+y+")").call(s).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}c(v,"makeGrid");function d(f,y){let g=0,T=Object.keys(z).map(p=>[p,z[p]]);x.append("g").selectAll("text").data(T).enter().append(function(p){let o=p[0].split(le.lineBreakRegex),l=-(o.length-1)/2,h=S.createElementNS("http://www.w3.org/2000/svg","text");h.setAttribute("dy",l+"em");for(let[m,w]of o.entries()){let s=S.createElementNS("http://www.w3.org/2000/svg","tspan");s.setAttribute("alignment-baseline","central"),s.setAttribute("x","10"),m>0&&s.setAttribute("dy","1em"),s.textContent=w,h.appendChild(s)}return h}).attr("x",10).attr("y",function(p,o){if(o>0)for(let l=0;l<o;l++)return g+=T[o-1][1],p[1]*f/2+g*f+y;else return p[1]*f/2+y}).attr("font-size",r.sectionFontSize).attr("class",function(p){for(let[o,l]of W.entries())if(p[0]===l)return"sectionTitle sectionTitle"+o%r.numberSectionStyles;return"sectionTitle"})}c(d,"vertLabels");function M(f,y,g,T){let p=n.db.getTodayMarker();if(p==="off")return;let o=x.append("g").attr("class","today"),l=new Date,h=o.append("line");h.attr("x1",k(l)+f).attr("x2",k(l)+f).attr("y1",r.titleTopMargin).attr("y2",T-r.titleTopMargin).attr("class","today"),p!==""&&h.attr("style",p.replace(/,/g,";"))}c(M,"drawToday");function Y(f){let y={},g=[];for(let T=0,p=f.length;T<p;++T)Object.prototype.hasOwnProperty.call(y,f[T])||(y[f[T]]=!0,g.push(f[T]));return g}c(Y,"checkUnique")},"draw"),Ni={setConf:Wi,draw:Vi},Ri=c(t=>`
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
`,"getStyles"),zi=Ri,Ui={parser:Je,db:Oi,renderer:Ni,styles:zi};export{Ui as diagram};

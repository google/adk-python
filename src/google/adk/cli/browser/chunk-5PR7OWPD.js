import{a as D}from"./chunk-DMWOYWYQ.js";import{a as z}from"./chunk-T3Q3QCCV.js";import"./chunk-NQKWI5EB.js";import"./chunk-WR6HISGZ.js";import"./chunk-I4UDKKN5.js";import"./chunk-2DLZXFEQ.js";import"./chunk-JNY2YWG7.js";import"./chunk-XULIXUQL.js";import"./chunk-3NJNOY56.js";import"./chunk-NALL4A3P.js";import{a as _}from"./chunk-TPDTRWWV.js";import{l as y}from"./chunk-WBLSVR3V.js";import"./chunk-GP6TCC26.js";import{A as w,N as T,R as S,S as O,T as k,U as R,V as I,W as E,X as F,p as A,r as L}from"./chunk-QFMJV7VH.js";import{g as o,i as M}from"./chunk-JRNAXTJ7.js";import"./chunk-ASJUXEUE.js";import"./chunk-EGBSMT36.js";import{a as C,j as b}from"./chunk-RMXJBC7V.js";var x={showLegend:!0,ticks:5,max:null,min:0,graticule:"circle"},G={axes:[],curves:[],options:x},g=structuredClone(G),U=L.radar,X=o(()=>y(C(C({},U),w().radar)),"getConfig"),P=o(()=>g.axes,"getAxes"),Y=o(()=>g.curves,"getCurves"),Z=o(()=>g.options,"getOptions"),q=o(a=>{g.axes=a.map(t=>({name:t.name,label:t.label??t.name}))},"setAxes"),J=o(a=>{g.curves=a.map(t=>({name:t.name,label:t.label??t.name,entries:K(t.entries)}))},"setCurves"),K=o(a=>{if(a[0].axis==null)return a.map(e=>e.value);let t=P();if(t.length===0)throw new Error("Axes must be populated before curves for reference entries");return t.map(e=>{let r=a.find(n=>n.axis?.$refText===e.name);if(r===void 0)throw new Error("Missing entry for axis "+e.label);return r.value})},"computeCurveEntries"),Q=o(a=>{let t=a.reduce((e,r)=>(e[r.name]=r,e),{});g.options={showLegend:t.showLegend?.value??x.showLegend,ticks:t.ticks?.value??x.ticks,max:t.max?.value??x.max,min:t.min?.value??x.min,graticule:t.graticule?.value??x.graticule}},"setOptions"),tt=o(()=>{S(),g=structuredClone(G)},"clear"),$={getAxes:P,getCurves:Y,getOptions:Z,setAxes:q,setCurves:J,setOptions:Q,getConfig:X,clear:tt,setAccTitle:O,getAccTitle:k,setDiagramTitle:E,getDiagramTitle:F,getAccDescription:I,setAccDescription:R},et=o(a=>{D(a,$);let{axes:t,curves:e,options:r}=a;$.setAxes(t),$.setCurves(e),$.setOptions(r)},"populate"),at={parse:o(a=>b(null,null,function*(){let t=yield z("radar",a);M.debug(t),et(t)}),"parse")},rt=o((a,t,e,r)=>{let n=r.db,i=n.getAxes(),l=n.getCurves(),s=n.getOptions(),c=n.getConfig(),d=n.getDiagramTitle(),p=_(t),u=nt(p,c),m=s.max??Math.max(...l.map(f=>Math.max(...f.entries))),h=s.min,v=Math.min(c.width,c.height)/2;st(u,i,v,s.ticks,s.graticule),ot(u,i,v,c),W(u,i,l,h,m,s.graticule,c),H(u,l,s.showLegend,c),u.append("text").attr("class","radarTitle").text(d).attr("x",0).attr("y",-c.height/2-c.marginTop)},"draw"),nt=o((a,t)=>{let e=t.width+t.marginLeft+t.marginRight,r=t.height+t.marginTop+t.marginBottom,n={x:t.marginLeft+t.width/2,y:t.marginTop+t.height/2};return T(a,r,e,t.useMaxWidth??!0),a.attr("viewBox",`0 0 ${e} ${r}`),a.append("g").attr("transform",`translate(${n.x}, ${n.y})`)},"drawFrame"),st=o((a,t,e,r,n)=>{if(n==="circle")for(let i=0;i<r;i++){let l=e*(i+1)/r;a.append("circle").attr("r",l).attr("class","radarGraticule")}else if(n==="polygon"){let i=t.length;for(let l=0;l<r;l++){let s=e*(l+1)/r,c=t.map((d,p)=>{let u=2*p*Math.PI/i-Math.PI/2,m=s*Math.cos(u),h=s*Math.sin(u);return`${m},${h}`}).join(" ");a.append("polygon").attr("points",c).attr("class","radarGraticule")}}},"drawGraticule"),ot=o((a,t,e,r)=>{let n=t.length;for(let i=0;i<n;i++){let l=t[i].label,s=2*i*Math.PI/n-Math.PI/2;a.append("line").attr("x1",0).attr("y1",0).attr("x2",e*r.axisScaleFactor*Math.cos(s)).attr("y2",e*r.axisScaleFactor*Math.sin(s)).attr("class","radarAxisLine"),a.append("text").text(l).attr("x",e*r.axisLabelFactor*Math.cos(s)).attr("y",e*r.axisLabelFactor*Math.sin(s)).attr("class","radarAxisLabel")}},"drawAxes");function W(a,t,e,r,n,i,l){let s=t.length,c=Math.min(l.width,l.height)/2;e.forEach((d,p)=>{if(d.entries.length!==s)return;let u=d.entries.map((m,h)=>{let v=2*Math.PI*h/s-Math.PI/2,f=B(m,r,n,c),j=f*Math.cos(v),N=f*Math.sin(v);return{x:j,y:N}});i==="circle"?a.append("path").attr("d",V(u,l.curveTension)).attr("class",`radarCurve-${p}`):i==="polygon"&&a.append("polygon").attr("points",u.map(m=>`${m.x},${m.y}`).join(" ")).attr("class",`radarCurve-${p}`)})}o(W,"drawCurves");function B(a,t,e,r){let n=Math.min(Math.max(a,t),e);return r*(n-t)/(e-t)}o(B,"relativeRadius");function V(a,t){let e=a.length,r=`M${a[0].x},${a[0].y}`;for(let n=0;n<e;n++){let i=a[(n-1+e)%e],l=a[n],s=a[(n+1)%e],c=a[(n+2)%e],d={x:l.x+(s.x-i.x)*t,y:l.y+(s.y-i.y)*t},p={x:s.x-(c.x-l.x)*t,y:s.y-(c.y-l.y)*t};r+=` C${d.x},${d.y} ${p.x},${p.y} ${s.x},${s.y}`}return`${r} Z`}o(V,"closedRoundCurve");function H(a,t,e,r){if(!e)return;let n=(r.width/2+r.marginRight)*3/4,i=-(r.height/2+r.marginTop)*3/4,l=20;t.forEach((s,c)=>{let d=a.append("g").attr("transform",`translate(${n}, ${i+c*l})`);d.append("rect").attr("width",12).attr("height",12).attr("class",`radarLegendBox-${c}`),d.append("text").attr("x",16).attr("y",0).attr("class","radarLegendText").text(s.label)})}o(H,"drawLegend");var it={draw:rt},lt=o((a,t)=>{let e="";for(let r=0;r<a.THEME_COLOR_LIMIT;r++){let n=a[`cScale${r}`];e+=`
		.radarCurve-${r} {
			color: ${n};
			fill: ${n};
			fill-opacity: ${t.curveOpacity};
			stroke: ${n};
			stroke-width: ${t.curveStrokeWidth};
		}
		.radarLegendBox-${r} {
			fill: ${n};
			fill-opacity: ${t.curveOpacity};
			stroke: ${n};
		}
		`}return e},"genIndexStyles"),ct=o(a=>{let t=A(),e=w(),r=y(t,e.themeVariables),n=y(r.radar,a);return{themeVariables:r,radarOptions:n}},"buildRadarStyleOptions"),dt=o(({radar:a}={})=>{let{themeVariables:t,radarOptions:e}=ct(a);return`
	.radarTitle {
		font-size: ${t.fontSize};
		color: ${t.titleColor};
		dominant-baseline: hanging;
		text-anchor: middle;
	}
	.radarAxisLine {
		stroke: ${e.axisColor};
		stroke-width: ${e.axisStrokeWidth};
	}
	.radarAxisLabel {
		dominant-baseline: middle;
		text-anchor: middle;
		font-size: ${e.axisLabelFontSize}px;
		color: ${e.axisColor};
	}
	.radarGraticule {
		fill: ${e.graticuleColor};
		fill-opacity: ${e.graticuleOpacity};
		stroke: ${e.graticuleColor};
		stroke-width: ${e.graticuleStrokeWidth};
	}
	.radarLegendText {
		text-anchor: start;
		font-size: ${e.legendFontSize}px;
		dominant-baseline: hanging;
	}
	${lt(t,e)}
	`},"styles"),vt={parser:at,db:$,renderer:it,styles:dt};export{vt as diagram};

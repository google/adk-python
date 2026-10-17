import{a as P}from"./chunk-CDENLQJG.js";import{a as z}from"./chunk-DUIJAVI2.js";import"./chunk-WKUOFYJM.js";import"./chunk-Q2HGE4NL.js";import"./chunk-LRCABWFD.js";import"./chunk-HKS4B5E7.js";import"./chunk-MM6RFXD4.js";import"./chunk-CUFJZGL2.js";import"./chunk-VWXZP4RG.js";import"./chunk-LTLDYHLB.js";import"./chunk-3D7C5O5T.js";import"./chunk-RKQR7KXB.js";import"./chunk-7YSH3HE7.js";import"./chunk-XNKUIZIT.js";import"./chunk-ARAETZM4.js";import"./chunk-DILFQQVU.js";import"./chunk-B5D7FH62.js";import"./chunk-6XZG3KG5.js";import{a as D}from"./chunk-IDCXNWW3.js";import{o as y}from"./chunk-TLHHCYOY.js";import"./chunk-ROUF34MA.js";import{O as S,T as k,U as O,V as R,W as I,X as _,Y as E,Z as F,i as L,k as T,u as A}from"./chunk-BTIP5KPZ.js";import{g as b}from"./chunk-JHJZJ2D5.js";import{a as i}from"./chunk-YUSHYV7C.js";import{a as C,g as M}from"./chunk-BZGM3F3K.js";var x={showLegend:!0,ticks:5,max:null,min:0,graticule:"circle"},w=32,G={axes:[],curves:[],options:x},g=structuredClone(G),K=T.radar,N=i(()=>y(C(C({},K),A().radar)),"getConfig"),B=i(()=>g.axes,"getAxes"),Y=i(()=>g.curves,"getCurves"),Z=i(()=>g.options,"getOptions"),q=i(a=>{g.axes=a.map(t=>({name:t.name,label:t.label??t.name}))},"setAxes"),J=i(a=>{g.curves=a.map(t=>({name:t.name,label:t.label??t.name,entries:Q(t.entries)}))},"setCurves"),Q=i(a=>{if(a[0].axis==null)return a.map(e=>e.value);let t=B();if(t.length===0)throw new Error("Axes must be populated before curves for reference entries");return t.map(e=>{let r=a.find(n=>n.axis?.$refText===e.name);if(r===void 0)throw new Error("Missing entry for axis "+e.label);return r.value})},"computeCurveEntries"),tt=i(a=>{let t=a.reduce((e,r)=>(e[r.name]=r,e),{});g.options={showLegend:t.showLegend?.value??x.showLegend,ticks:t.ticks?.value??x.ticks,max:t.max?.value??x.max,min:t.min?.value??x.min,graticule:t.graticule?.value??x.graticule},g.options.ticks>w&&(b.warn(`Radar diagram ticks (${g.options.ticks}) exceeds maximum allowed (${w}). Using ${w} instead.`),g.options.ticks=w)},"setOptions"),et=i(()=>{k(),g=structuredClone(G)},"clear"),$={getAxes:B,getCurves:Y,getOptions:Z,setAxes:q,setCurves:J,setOptions:tt,getConfig:N,clear:et,setAccTitle:O,getAccTitle:R,setDiagramTitle:E,getDiagramTitle:F,getAccDescription:_,setAccDescription:I},at=i(a=>{P(a,$);let{axes:t,curves:e,options:r}=a;$.setAxes(t),$.setCurves(e),$.setOptions(r)},"populate"),rt={parse:i(a=>M(null,null,function*(){let t=yield z("radar",a);b.debug(t),at(t)}),"parse")},nt=i((a,t,e,r)=>{let n=r.db,l=n.getAxes(),c=n.getCurves(),s=n.getOptions(),o=n.getConfig(),d=n.getDiagramTitle(),p=D(t),u=st(p,o),m=s.max??Math.max(...c.map(f=>Math.max(...f.entries))),h=s.min,v=Math.min(o.width,o.height)/2;ot(u,l,v,s.ticks,s.graticule),it(u,l,v,o),W(u,l,c,h,m,s.graticule,o),j(u,c,s.showLegend,o),u.append("text").attr("class","radarTitle").text(d).attr("x",0).attr("y",-o.height/2-o.marginTop)},"draw"),st=i((a,t)=>{let e=t.width+t.marginLeft+t.marginRight,r=t.height+t.marginTop+t.marginBottom,n={x:t.marginLeft+t.width/2,y:t.marginTop+t.height/2};return S(a,r,e,t.useMaxWidth??!0),a.attr("viewBox",`0 0 ${e} ${r}`).attr("overflow","visible"),a.append("g").attr("transform",`translate(${n.x}, ${n.y})`)},"drawFrame"),ot=i((a,t,e,r,n)=>{if(n==="circle")for(let l=0;l<r;l++){let c=e*(l+1)/r;a.append("circle").attr("r",c).attr("class","radarGraticule")}else if(n==="polygon"){let l=t.length;for(let c=0;c<r;c++){let s=e*(c+1)/r,o=t.map((d,p)=>{let u=2*p*Math.PI/l-Math.PI/2,m=s*Math.cos(u),h=s*Math.sin(u);return`${m},${h}`}).join(" ");a.append("polygon").attr("points",o).attr("class","radarGraticule")}}},"drawGraticule"),it=i((a,t,e,r)=>{let n=t.length;for(let l=0;l<n;l++){let c=t[l].label,s=2*l*Math.PI/n-Math.PI/2,o=Math.cos(s),d=Math.sin(s);a.append("line").attr("x1",0).attr("y1",0).attr("x2",e*r.axisScaleFactor*o).attr("y2",e*r.axisScaleFactor*d).attr("class","radarAxisLine");let p=o>.01?"start":o<-.01?"end":"middle",u=d>.01?"hanging":d<-.01?"auto":"central",m=4;a.append("text").text(c).attr("x",e*r.axisLabelFactor*o+m*o).attr("y",e*r.axisLabelFactor*d+m*d).attr("text-anchor",p).attr("dominant-baseline",u).attr("class","radarAxisLabel")}},"drawAxes");function W(a,t,e,r,n,l,c){let s=t.length,o=Math.min(c.width,c.height)/2;e.forEach((d,p)=>{if(d.entries.length!==s)return;let u=d.entries.map((m,h)=>{let v=2*Math.PI*h/s-Math.PI/2,f=V(m,r,n,o),U=f*Math.cos(v),X=f*Math.sin(v);return{x:U,y:X}});l==="circle"?a.append("path").attr("d",H(u,c.curveTension)).attr("class",`radarCurve-${p}`):l==="polygon"&&a.append("polygon").attr("points",u.map(m=>`${m.x},${m.y}`).join(" ")).attr("class",`radarCurve-${p}`)})}i(W,"drawCurves");function V(a,t,e,r){let n=Math.min(Math.max(a,t),e);return r*(n-t)/(e-t)}i(V,"relativeRadius");function H(a,t){let e=a.length,r=`M${a[0].x},${a[0].y}`;for(let n=0;n<e;n++){let l=a[(n-1+e)%e],c=a[n],s=a[(n+1)%e],o=a[(n+2)%e],d={x:c.x+(s.x-l.x)*t,y:c.y+(s.y-l.y)*t},p={x:s.x-(o.x-c.x)*t,y:s.y-(o.y-c.y)*t};r+=` C${d.x},${d.y} ${p.x},${p.y} ${s.x},${s.y}`}return`${r} Z`}i(H,"closedRoundCurve");function j(a,t,e,r){if(!e)return;let n=(r.width/2+r.marginRight)*3/4,l=-(r.height/2+r.marginTop)*3/4,c=20;t.forEach((s,o)=>{let d=a.append("g").attr("transform",`translate(${n}, ${l+o*c})`);d.append("rect").attr("width",12).attr("height",12).attr("class",`radarLegendBox-${o}`),d.append("text").attr("x",16).attr("y",0).attr("class","radarLegendText").text(s.label)})}i(j,"drawLegend");var lt={draw:nt},ct=i((a,t)=>{let e="";for(let r=0;r<a.THEME_COLOR_LIMIT;r++){let n=a[`cScale${r}`];e+=`
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
		`}return e},"genIndexStyles"),dt=i(a=>{let t=L(),e=A(),r=y(t,e.themeVariables),n=y(r.radar,a);return{themeVariables:r,radarOptions:n}},"buildRadarStyleOptions"),ut=i(({radar:a}={})=>{let{themeVariables:t,radarOptions:e}=dt(a);return`
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
	${ct(t,e)}
	`},"styles"),ft={parser:rt,db:$,renderer:lt,styles:ut};export{ft as diagram};

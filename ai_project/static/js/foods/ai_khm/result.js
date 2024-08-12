function showFoodMap(lat, lng) {
	lat = Number(lat);
	lng = Number(lng);

	const container = document.getElementById('foodMap');
	const options = {
		center: new kakao.maps.LatLng(lat, lng),
		level: 3
	};

	let map = new kakao.maps.Map(container, options);

	const markerPosition  = new kakao.maps.LatLng(lat, lng); 

	const marker = new kakao.maps.Marker({
		position: markerPosition
	});

	marker.setMap(map);
}

// 밑의 로직은 처음에 첫번째 목록 불러오는 로직
const data = document.currentScript.dataset;
const gLat1 = data.lat1;
const gLng2 = data.lng1;

const gContainer = document.getElementById('foodMap');
const gOptions = {
	center: new kakao.maps.LatLng(gLat1, gLng2),
	level: 3
};

let gMap = new kakao.maps.Map(gContainer, gOptions);

const gMarkerPosition  = new kakao.maps.LatLng(gLat1, gLng2); 

const gMarker = new kakao.maps.Marker({
	position: gMarkerPosition
});

gMarker.setMap(gMap);